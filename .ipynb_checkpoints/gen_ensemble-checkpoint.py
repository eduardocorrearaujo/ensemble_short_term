import numpy as np
import pandas as pd
import matplotlib as mpl
from epiweeks import Week
import matplotlib.pyplot as plt 
from scipy.special import inv_boxcox 
from mosqlient.forecast import Ensemble

# Definir a cor das bordas (spines) como cinza
mpl.rcParams['axes.edgecolor'] = 'gray'

# Definir a cor das linhas dos ticks maiores e menores como cinza
mpl.rcParams['xtick.color'] = 'gray'
mpl.rcParams['ytick.color'] = 'gray'
mpl.rcParams['xtick.labelcolor'] = 'black'
mpl.rcParams['ytick.labelcolor'] = 'black'

states_BR = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI',
 'RN', 'MG', 'RJ', 'ES', 'AM', 'AP',
 'RR', 'RO', 'AC', 'PA', 'DF', 'GO', 'MT', 'MS',
 'RS', 'SC', 'PR', 'BR', 'SE', 'SP', 'TO'] 



def get_preds(state, model, week): 

    df = pd.read_csv(f'forecast_tables/for_{model}_se_{week}_{state}.csv.gz')
 
    df.date = pd.to_datetime(df.date)
        
    return df

def get_all_preds(state, week):
    if week ==0:
        week = 52

    df_pred1 = get_preds(state, 'lstm', week = week)
    df_pred1['model_id'] = 1
    
    df_pred2 = get_preds(state, 'gp', week = week)
    df_pred2['model_id'] = 2

    df_pred3 = get_preds(state, 'arima', week = week)
    df_pred3['model_id'] = 3

    df_preds =pd.concat([df_pred1, df_pred2, df_pred3])
    
    return df_preds.rename(columns = {'lower': 'lower_95', 
                                         'upper': 'upper_95'})
    
def get_ensemble(state, max_date, week):

    df = pd.read_csv(f'data/dengue_{state}.csv.gz')
    df.date = pd.to_datetime(df.date)
    df = df.loc[df.date == max_date]
    
    df['casos'] = inv_boxcox(df['casos'].values[0], 0.05) -1 

    df_preds = get_all_preds(state= state, week= for_week-1)

    df_preds = df_preds.loc[df_preds.date == max_date]

    e1 = Ensemble(df = df_preds,
            order_models = [1, 2, 3], 
            dist = 'log_normal',
            mixture = 'log',
            fn_loss = 'median', 
            conf_level = 0.95)
    
    weights_crps = e1.compute_weights(df, metric= 'crps') 

    df_for = get_all_preds(state= state, week= for_week)

    efor = Ensemble(df = df_for,
        order_models = [1, 2, 3], 
        dist = 'log_normal',
        mixture = 'log',
        fn_loss = 'median', 
        conf_level = 0.95)

    weights = weights_crps['weights']

    df_ens_crps = efor.apply_ensemble(weights=weights,
                                      p=np.array([0.025, 0.5, 0.975]))
    

    return df_ens_crps, weights_crps 


def make_plot(state, for_week, df_crps, df_ens_):
        
    data = pd.read_csv(f'data/dengue_{state}.csv.gz')
        
    data.date = pd.to_datetime(data.date)

    data = data.sort_values(by = 'date')
    
    data = data.tail(12)
    
    data['casos'] = inv_boxcox(data['casos'].values, 0.05) -1
    
    df_ens_crps = df_crps.loc[df_crps.state == state]
    #df_ens = df_ens_.loc[df_ens_.state == state].reset_index(drop=True)
    if state !='BR':
        df_ens = df_ens_.loc[df_ens_.state == state].reset_index(drop=True)
    else:
        df_ens = df_ens_
        
    fig,ax = plt.subplots(1,2, figsize = (13, 5))
    
    for ax_ in ax.ravel(): 
        ax_.plot(data.date,  data.casos, color = 'black', linewidth = 2, linestyle = '-', label = 'Casos')
        
        ax_.plot([data.date.values[-1], df_ens_crps.date.values[0]], [data['casos'].values[-1], df_ens_crps.pred.values[0]], ls = '--', color = 'black')
        
        ax_.plot(df_ens_crps.date,  df_ens_crps.pred, color = 'tab:red', label = 'Forecast curto prazo')
                
        ax_.fill_between(df_ens_crps.date, df_ens_crps.lower, df_ens_crps.upper, color = 'tab:red', alpha = 0.1)
    
        ax_.plot(df_ens.date, df_ens.pred_ensemble_23, color = '#35DB65', label = 'Ensemble (2023)')
        
        ax_.plot(df_ens.date, df_ens.pred_ensemble_24, color = '#3577DB', label = 'Ensemble (2024)')

        if state != 'BR':
            ax_.fill_between(df_ens.date, df_ens.lower_ensemble_23, df_ens.upper_ensemble_23, color = '#35DB65', alpha = 0.2)
                                
            ax_.fill_between(df_ens.date, df_ens.lower_ensemble_24, df_ens.upper_ensemble_24, color = '#3577DB', alpha = 0.2)
            
    
    for ax_ in ax.ravel():
        ax_.set_xlabel('Data')
        ax_.set_ylabel('Novos Casos')
        ax_.set_title(f'Forecast casos prováveis - {state}')
        ax_.legend()
        ax_.grid()
    
    ax[0].set_ylim(min(0.9*min(data.casos), (min(df_ens_crps.lower))), max(1.25*max(df_ens_crps.upper), max(data.casos)))
    
    fig.autofmt_xdate(rotation=30, ha='center')
        
    
    plt.savefig(f'./figures/forecast_{for_week}_{state}.png', dpi = 300, bbox_inches = 'tight')
    plt.savefig(f'./figures/forecast_{state}.png', dpi = 300, bbox_inches = 'tight')
    plt.close()


def make_plot_new(state, for_week, df_crps):
        
    data = pd.read_csv(f'data/dengue_{state}.csv.gz')
        
    data.date = pd.to_datetime(data.date)

    data = data.sort_values(by = 'date')
    
    data = data.tail(12)
    
    data['casos'] = inv_boxcox(data['casos'].values, 0.05) -1
    
    df_ens_crps = df_crps.loc[df_crps.state == state]

    fig,ax = plt.subplots(1, figsize = (6.5, 5))
    
    ax.plot(data.date,  data.casos, color = 'black', linewidth = 2, linestyle = '-', label = 'Casos')
        
    ax.plot([data.date.values[-1], df_ens_crps.date.values[0]], [data['casos'].values[-1], df_ens_crps.pred.values[0]], ls = '--', color = 'black')
        
    ax.plot(df_ens_crps.date,  df_ens_crps.pred, color = 'tab:red', label = 'Forecast curto prazo')
                
    ax.fill_between(df_ens_crps.date, df_ens_crps.lower_95, df_ens_crps.upper_95, color = 'tab:red', alpha = 0.1)
           
    ax.set_xlabel('Data')
    ax.set_ylabel('Novos Casos')
    ax.set_title(f'Forecast casos prováveis - {state}')
    ax.legend()
    ax.grid()
        
    fig.autofmt_xdate(rotation=30, ha='center')
        
    #plt.savefig(f'./figures/forecast_{for_week}_{state}.png', dpi = 300, bbox_inches = 'tight')
    plt.savefig(f'./figures/forecast_{state}.png', dpi = 300, bbox_inches = 'tight')
    plt.savefig(f'./figures/forecast_{state}.svg', format="svg", bbox_inches = 'tight')
    plt.close()

if __name__ == '__main__':

    df = pd.read_csv('data/dengue_BR.csv.gz')
    df.date = pd.to_datetime(df.date)
    max_date = df.date.max()
    epi_week = Week.fromdate(df.date.max())
    for_week = epi_week.week
    year = epi_week.year

    # compute ensemble
    df_crps = pd.DataFrame()
    #df_w = pd.DataFrame()

    for state in states_BR:
        print(state)
        df_crps_ , weights_ = get_ensemble(state, max_date, for_week)
        df_crps_['state'] = state
        df_crps = pd.concat([df_crps, df_crps_], ignore_index = True)

        #df_w_ = pd.DataFrame(weights_, columns = ['weights'])
        #df_w_['model']  = ['model_1', 'model_2', 'model_3']
        #df_w_['state'] = state
        #df_w = pd.concat([df_w, df_w_])

    df_crps.to_csv(f'forecast_tables/for_ensemble_{year}_{for_week}.csv', index = False)
    df_crps.to_csv(f'forecast_tables/for_ensemble.csv', index = False)


    for state in states_BR:
        make_plot_new(state, for_week, df_crps)
