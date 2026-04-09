import pandas as pd 
import model_gp as gp
import model_arima as ar 
import model_lstm as lstm 

states_BR = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI',
 'SE', 'RN', 'SP', 'MG', 'RJ', 'ES', 'AM', 'AP', 'TO',
 'RR', 'RO', 'AC', 'PA', 'DF', 'GO', 'MT', 'MS',
 'RS', 'SC', 'PR', 'BR'] 

disease = 'chik'
train_ini_date = '2023-06-01'
end_train_date = '2025-08-01'

df = pd.read_csv(f'data/{disease}_update.csv.gz', index_col = 'Unnamed: 0')

df.date = pd.to_datetime(df.date)

df.set_index('date', inplace = True)

df = df.groupby('uf').resample('W-SUN').sum().drop(['uf'], axis = 1).reset_index()

df = df.rename(columns = {'date': 'dates',
                            'casos': 'y'})

for state in ['GO']:
    print(state)
    # input to arima model
    #print('--------------------- Training ARIMA ---------------------')

    #ar.train_model(df, state, train_ini_date=train_ini_date, 
    #               train_end_date = end_train_date, disease=disease)

    # train gpr model 
    #print('--------------------- Training GP ---------------------')

    #gp.train_model(state, ini_train=train_ini_date, end_train = end_train_date, disease=disease)

    # train lstm model
    
    print('--------------------- Training LSTM ---------------------')

    df_ = pd.read_csv(f'data/{disease}_{state}.csv.gz')

    feat = df_.shape[1]-1
    HIDDEN = 64
    LOOK_BACK = 8
    PREDICT_N = 3

    model = lstm.build_lstm(hidden=HIDDEN, features=feat, predict_n=PREDICT_N, look_back=LOOK_BACK,
                            batch_size=1, loss='msle')


    model.compile(loss='msle', optimizer='adam', metrics=["accuracy", "mape", "mse"])
        
    lstm.train_model(model, state, doenca=disease,
                    end_train_date=None,
                    ratio = 1,
                    ini_date = train_ini_date,
                    end_date = end_train_date,
                    filename=f'data/{disease}_{state}.csv.gz',
                    min_delta=0.001/2, label='state',
                    patience = 30, 
                    epochs=400,
                    batch_size=1,
                    predict_n=PREDICT_N,
                    look_back=LOOK_BACK)

    