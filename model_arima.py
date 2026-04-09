import joblib
import pickle
import pandas as pd 
from model_gp import get_next_n_weeks
from pmdarima import preprocessing as ppc
from mosqlient.forecast import Arima
from epiweeks import Week 

def get_prediction_dataframe(preds, date, boxcox) -> pd.DataFrame:
    """
    Function to organize the predictions of the ARIMA model in a pandas DataFrame.

    Parameters
    ----------
    horizon: int
        The number of weeks forecasted by the model
    end_date: str
        Last week of the out of the sample evaluation. The first week is after the last training observation.
    plot: bool
        If true the plot of the model out of the sample is returned
    """

    df_preds = pd.DataFrame()

    df_preds["date"] = date

    try:
        df_preds["pred"] = preds[0].values

    except:
        df_preds["pred"] = preds[0]

    df_preds.loc[:, ["lower", "upper"]] = preds[1]

    if df_preds["pred"].values[0] == 0:
        df_preds = df_preds.iloc[1:]

    df_preds["pred"] = boxcox.inverse_transform(df_preds["pred"])[0]
    df_preds["lower"] = boxcox.inverse_transform(df_preds["lower"])[0]
    df_preds["upper"] = boxcox.inverse_transform(df_preds["upper"])[0]

    return df_preds

def train_model(df, state, train_ini_date, train_end_date, disease):
    '''
    Function to train and save the arima model 
    '''
    if state != 'BR':
        df_ = df.loc[df.uf == state].drop(['uf'], axis =1 ).set_index('dates')
    else: 
        df_ = df.set_index('dates').drop(['uf'], axis =1).resample('W-SUN').sum()#.reset_index()

    df_['y'] = df_['y'] + 0.1

    m_arima = Arima(df = df_)

    model = m_arima.train( train_ini_date=train_ini_date, train_end_date = train_end_date)

    # Save model
    with open(f'saved_models/arima_{disease}_{state}.pkl', 'wb') as pkl:
        pickle.dump(model, pkl)
    
    # save transf on data
    bc_transformer = m_arima.boxcox
    joblib.dump(bc_transformer, f'saved_models/bc_{disease}_{state}.pkl')

def apply_model(df,state, disease):
    '''
    Function to load and apply the pre trained model 
    '''

    if state != 'BR':
        df_ = df.loc[df.uf == state].drop(['uf'], axis =1 ).set_index('dates')
    else: 
        df_ = df.set_index('dates').drop(['uf'], axis =1).resample('W-SUN').sum()#.reset_index()

    for_week = Week.fromdate(pd.to_datetime(df_.index.max()))

    df_['y'] = df_['y'] + 0.1

    bc = joblib.load(f'saved_models/bc_{disease}_{state}.pkl')

    df_.loc[:, "y"] = bc.transform(df_.y)[0]

    with open(f'saved_models/arima_{disease}_{state}.pkl', 'rb') as pkl:
        m_arima = pickle.load(pkl)

    # update the model with the new data:
    m_arima.update(df_)

    date = get_next_n_weeks(df_.index[-1].strftime("%Y-%m-%d"), 3)

    preds = m_arima.predict(3, return_conf_int=True)

    df_for = get_prediction_dataframe(preds, date, bc)

    df_for.to_csv(f'forecast_tables/for_arima_{disease}_{for_week.year}_{for_week.week}_{state}.csv.gz', index = False)

    return 

