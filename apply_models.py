import numpy as np
import pandas as pd 
import model_gp as gp
import model_arima as ar 
import model_lstm as lstm 
from epiweeks import Week

states_BR = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI',
 'SE', 'RN', 'SP', 'MG', 'RJ', 'ES', 'AM', 'AP', 'TO',
 'RR', 'RO', 'AC', 'PA', 'DF', 'GO', 'MT', 'MS',
 'RS', 'SC', 'PR', 'BR']

disease = 'dengue'

#epiweek = '202541'


for epiweek in ['202546']: 

    if epiweek is None: 
        df = pd.read_csv(f'data/{disease}_BR.csv.gz')
    else: 
        df = pd.read_csv(f'data/{disease}_BR_{epiweek}.csv.gz')
                        
    df.date = pd.to_datetime(df.date)
    end_date = df.date.max().strftime('%Y-%m-%d')
    for_week = Week.fromdate(df.date.max())

    for state in states_BR:

        if epiweek is None: 
            df = pd.read_csv(f'data/{disease}_update.csv.gz', index_col = 'Unnamed: 0')

        else: 
            df = pd.read_csv(f'data/{disease}_{epiweek}_update.csv.gz', index_col = 'Unnamed: 0')

        df.date = pd.to_datetime(df.date)

        df = df.loc[df.date <= end_date]

        df.set_index('date', inplace = True)

        df = df.groupby('uf').resample('W-SUN').sum().drop(['uf'], axis = 1).reset_index()

        df = df.rename(columns = {'date': 'dates',
                                    'casos': 'y'})
        
        ar.apply_model(df, state, disease = disease)

        #print('--------------------- Apply GP ---------------------')

        gp.apply_model(state, end_date, disease = disease, epiweek=epiweek)
            
        #print('--------------------- Apply LSTM ---------------------')

        if epiweek is None: 
            FILENAME_DATA = f'data/{disease}_{state}.csv.gz'
        else: 
            FILENAME_DATA = f'data/{disease}_{state}_{epiweek}.csv.gz'

        df_ = pd.read_csv(FILENAME_DATA, index_col = 'date')

        feat = df_.shape[1]
            
        model_name = f'trained_{state}_{disease}_state'

        print(model_name)

        df_for, X_for = lstm.apply_forecast(state, None, end_date, look_back=4, predict_n=3,
                                            filename=FILENAME_DATA, model_name=model_name)

        #print(X_for)
        
        df_for.date = pd.to_datetime(df_for.date)

        df_for.to_csv(f'forecast_tables/for_lstm_{disease}_{for_week.year}_{for_week.week}_{state}.csv.gz', index = False)
