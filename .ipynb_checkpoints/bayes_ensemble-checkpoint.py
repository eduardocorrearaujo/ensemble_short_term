import numpy as np
import pandas as pd
from epiweeks import Week
from scipy.stats import lognorm
from scipy.special import logsumexp
from scipy.optimize import minimize
from scoringrules import crps_lognormal


def invlogit(y):
    return 1 / (1 + np.exp(-y))

def alpha_01(alpha_inv):
    K = len(alpha_inv) + 1
    z = np.full(K-1, np.nan)  # Equivalent to rep(NA, K-1)
    alphas = np.zeros(K)      # Equivalent to rep(0, K)
    
    for k in range(K-1):
        z[k] = invlogit(alpha_inv[k] + np.log(1 / (K - k)))
        alphas[k] = (1 - np.sum(alphas[:k])) * z[k]
    
    alphas[K-1] = 1 - np.sum(alphas[:-1])
    return alphas


def pool_par_gauss(alpha, m, v):
    ws = alpha / v
    vstar = 1 / np.sum(ws)
    mstar = np.sum(ws * m) * vstar
    return mstar, np.sqrt(vstar)
    
def get_lognormal_pars(med, lwr, upr, alpha=0.90):
    def loss2(theta):
        tent_qs = lognorm.ppf([(1 - alpha)/2, (1 + alpha)/2], s=theta[1], scale=np.exp(theta[0]))
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr
        return attained_loss

    mustar = np.log(med)
    result = minimize(loss2, x0=[mustar, 0.5], bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 10)],method = "Nelder-mead")
    return result.x

def get_lognormal_pool_pars(ms, vs, weights):
    pars = pool_par_gauss(alpha=weights, m=ms, v=vs)
    return pars

def get_df_log_pars(preds_, alpha = 0.9):
    compute_pars_result = compute_pars(preds_)
    par_df = pd.DataFrame(compute_pars_result, columns=["mu", "sigma"])
    
    # Combine the original preds and the computed parameters
    with_pars = pd.concat([preds_, par_df], axis=1)
    
    theo_preds_result = compute_theoretical_preds(par_df.to_numpy(), alpha)
    
    # Create a DataFrame for the theoretical predictions
    theo_pred_df = pd.DataFrame(theo_preds_result, columns=["fit_med", "fit_lwr", "fit_upr"])
    
    with_theo_preds = pd.concat([with_pars, theo_pred_df], axis=1)

    return with_theo_preds


def find_opt_LS_weights(obs, ms, vs):
    K = len(ms)
    if len(vs) != K:
        raise ValueError("ms and vs are not the same size!")

    def loss(eta):
        ws = alpha_01(eta)
        pool = get_lognormal_pool_pars(ms, vs, weights=ws)
        meanlog, sdlog = pool
        return -lognorm.logpdf(obs, s=sdlog, scale=np.exp(meanlog))

    initial_guess = np.random.normal(size=K - 1)#np.full(K-1, 1/(K-1))#
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }


def find_opt_LS_weights_all(obs, preds, order_models):
    '''
    obs: dataframe com colunas date and casos;
    ms: dataframe com colunas: date, mu, sigma, model_id
    '''
    K = len(order_models)

    def loss(eta):
        ws = alpha_01(eta)

        score = 0
        for date in obs.date:
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.sort_values(by = 'model_id')
            ms = preds_.mu,
            vs = preds_.sigma**2
            
            if len(vs) != K:
                raise ValueError("n_models and vs are not the same size!")
    
            pool = get_lognormal_pool_pars(ms, vs, weights=ws)
            meanlog, sdlog = pool
            score = score -lognorm.logpdf(obs, s=sdlog, scale=np.exp(meanlog))

        return score  

    initial_guess = np.random.normal(size=K - 1)#np.full(K-1, 1/(K-1))#
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }


def find_opt_CRPS_weights(obs, ms, vs):
    K = len(ms)
    if len(vs) != K:
        raise ValueError("ms and vs are not the same size!")

    def loss(eta):
        ws = alpha_01(eta)
        pool = get_lognormal_pool_pars(ms, vs, weights=ws)
        meanlog, sdlog = pool
        return crps_lognormal(observation = obs, mulog = meanlog, sigmalog = sdlog)

    initial_guess = np.random.normal(size=K - 1)#np.full(K-1, 1/(K-1))#
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }
    
def compute_pars(preds):
    return np.apply_along_axis(lambda row: get_lognormal_pars(med=row[0], lwr=row[1], upr=row[2]), 1, preds)


def compute_theoretical_preds(par_df, Alpha):
    return np.apply_along_axis(lambda row: lognorm.ppf([0.5, (1 - Alpha) / 2, (1 + Alpha) / 2], s=row[1], scale=np.exp(row[0])), 1, par_df)




def get_result(data, preds, alpha = 0.9, opt_weights = 'logscore'): 
    
    with_theo_preds = get_df_log_pars(preds, alpha = alpha) 

    K = preds.shape[0]
    alphas = np.full(K, 1/K)
    equal_pool = get_lognormal_pool_pars(ms=with_theo_preds["mu"].to_numpy(), vs=with_theo_preds["sigma"].to_numpy()**2, weights=alphas)
    
    # Calculate log scores
    omega = data
    logscores = lognorm.logpdf(x=omega, s=with_theo_preds["sigma"].to_numpy(), scale=np.exp(with_theo_preds["mu"].to_numpy()))

    df_w = pd.DataFrame()
    for i in np.arange(0,10):
        if opt_weights == 'logscore':
            LS_weights = find_opt_LS_weights(obs = omega,
                    ms = with_theo_preds.mu,
                    vs = with_theo_preds.sigma**2)
        if opt_weights == 'crps':
            LS_weights = find_opt_CRPS_weights(obs = omega,
                    ms = with_theo_preds.mu,
                    vs = with_theo_preds.sigma**2)
            
        df_ = pd.DataFrame(LS_weights)
        df_['run'] = i
        df_w = pd.concat([df_w, df_])
        
    df_w = df_w.loc[df_w.loss == df_w.loss.min()]
    df_w = df_w.loc[df_w.run == df_w.run.min()]

    LS_pool = pool_par_gauss(alpha = df_w['weights'].values, m = with_theo_preds.mu,
            v = with_theo_preds.sigma**2)

    p = np.array([0.5, 0.05, 0.95])

    # Calculate the quantiles using lognorm.ppf
    # The scale parameter is exp(meanlog) and the shape parameter is sdlog
    quantiles = lognorm.ppf(p, s=LS_pool[1], scale=np.exp(LS_pool[0]))

    df1 = pd.DataFrame([df_w['weights'].values], columns = preds.model_id.values,)
    
    df1.columns = 'weight_' + df1.columns.astype(str) 


    df2 = pd.DataFrame([quantiles], columns = ['pred', 'lower', 'upper'])

    df = pd.concat([df1,df2], axis =1)

    df.head()
    
    return df,df_w


def get_epiweek(date):
    '''
    Capturing the epidemiological year and week from the date 
    '''
    epiweek = Week.fromdate(date)
    return (epiweek.year, epiweek.week)


def apply_ensemble(weights, preds, alpha = 0.9): 
    
    with_theo_preds = get_df_log_pars(preds, alpha = alpha) 

    K = preds.shape[0]
    alphas = np.full(K, 1/K)
    equal_pool = get_lognormal_pool_pars(ms=with_theo_preds["mu"].to_numpy(), vs=with_theo_preds["sigma"].to_numpy()**2, weights=alphas)

    LS_pool = pool_par_gauss(alpha = weights, m = with_theo_preds.mu,
            v = with_theo_preds.sigma**2)

    p = np.array([0.5, 0.05, 0.95])

    # Calculate the quantiles using lognorm.ppf
    # The scale parameter is exp(meanlog) and the shape parameter is sdlog
    quantiles = lognorm.ppf(p, s=LS_pool[1], scale=np.exp(LS_pool[0]))

    df2 = pd.DataFrame([quantiles], columns = ['pred', 'lower', 'upper'])

    return df2

def dlnorm_mix(omega, mu, sigma, weights):

    lw = np.log(weights)
    K = len(mu)

    if len(sigma) != K:
        print('mu and sigma should be the same lenght')

    ldens = list(np.zeros(K))
    
    for i in np.arange(K):

        ldens[i] = lognorm.logpdf(omega, s=sigma, scale=np.exp(mu))

    return logsumexp(lw+ldens)

def crps_lognormal_mix(omega, mu, sigma, weights):

    K = len(mu)

    if len(sigma) != K:
        print('mu and sigma should be the same lenght')

    crpsdens = list(np.zeros(K))
    
    for i in np.arange(K):

        crpsdens[i] = crps_lognormal(observation = omega, mulog = mu[i], sigmalog = sigma[i])

    return np.dot(np.array(weights), np.array(crpsdens))#, crpsdens 


def get_forecast(weights, preds_25):
    df_for = pd.DataFrame()
    for d in preds_25.date.unique():
    
        preds_ = preds_25.loc[preds_25.date == d][['pred', 'lower', 'upper', 'date', 'model_id']].drop(['date'],axis =1).reset_index(drop = True)
        preds_ = preds_.sort_values(by = 'model_id')
        preds_ = get_df_log_pars(preds_)
    
        LS_pool = pool_par_gauss(alpha = weights, m = preds_.mu,
                v = preds_.sigma**2)
    
        p = np.array([0.5, 0.05, 0.95])
        
        # Calculate the quantiles using lognorm.ppf
        # The scale parameter is exp(meanlog) and the shape parameter is sdlog
        quantiles = lognorm.ppf(p, s=LS_pool[1], scale=np.exp(LS_pool[0]))
        
        df_ = pd.DataFrame([quantiles], columns = ['pred', 'lower', 'upper'])
    
        df_['date'] = d
        
        df_for = pd.concat([df_for, df_], axis =0).reset_index(drop = True)

    df_for.date = pd.to_datetime(df_for.date)
    
    return df_for
    

def find_opt_LS_weights_all(obs, preds, order_models):
    '''
    obs: dataframe com colunas date and casos;
    ms: dataframe com colunas: date, mu, sigma, model_id
    '''
    preds = get_df_log_pars(preds)
    
    K = len(order_models)

    def loss(eta):
        ws = alpha_01(eta)

        score = 0
        for date in obs.date:
            #print(date)
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.sort_values(by = 'model_id')
            preds_ = preds_.drop(['date'],axis =1).reset_index(drop = True)
            ms = preds_['mu']
          
            vs = preds_.sigma**2
   
            if len(vs) != K:
                print(date)
                raise ValueError("n_models and vs are not the same size!")
    
            pool = get_lognormal_pool_pars(ms, vs, weights=ws)
            meanlog, sdlog = pool
            score = score -lognorm.logpdf(obs.loc[obs.date == date].casos, s=sdlog, scale=np.exp(meanlog))

        return score  

    initial_guess = np.random.normal(size=K - 1)#np.full(K-1, 1/(K-1))#
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }


def find_opt_CRPS_weights_all(obs, preds, order_models):
    '''
    obs: dataframe com colunas date and casos;
    ms: dataframe com colunas: date, mu, sigma, model_id
    '''
    preds = get_df_log_pars(preds)
    
    K = len(order_models)

    def loss(eta):
        ws = alpha_01(eta)

        score = 0
        for date in obs.date:
            #print(date)
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.sort_values(by = 'model_id')
            preds_ = preds_.drop(['date'],axis =1).reset_index(drop = True)
            ms = preds_['mu']
          
            vs = preds_.sigma**2
   
            if len(vs) != K:
                print(date)
                raise ValueError("n_models and vs are not the same size!")
    
            pool = get_lognormal_pool_pars(ms, vs, weights=ws)
            meanlog, sdlog = pool
            score = score + crps_lognormal(observation = obs.loc[obs.date == date].casos, mulog = meanlog, sigmalog = sdlog)

        return score  

    initial_guess = np.full(K-1, 1/(K-1))#
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }



