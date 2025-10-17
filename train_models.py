import pandas as pd 
import model_gp as gp
import model_arima as ar 
import model_lstm as lstm 

states_BR = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI',
 'SE', 'RN', 'SP', 'MG', 'RJ', 'ES', 'AM', 'AP', 'TO',
 'RR', 'RO', 'AC', 'PA', 'DF', 'GO', 'MT', 'MS',
 'RS', 'SC', 'PR', 'BR'] 

end_train_date = '2025-01-12'

df = pd.read_csv('data/dengue_update.csv.gz', index_col = 'Unnamed: 0')

df.date = pd.to_datetime(df.date)

df.set_index('date', inplace = True)

df = df.groupby('uf').resample('W-SUN').sum().drop(['uf'], axis = 1).reset_index()

df = df.rename(columns = {'date': 'dates',
                            'casos': 'y'})

for state in states_BR:
    print(state)
    # input to arima model
    print('--------------------- Training ARIMA ---------------------')

    ar.train_model(df, state, train_end_date = end_train_date)

    # train gpr model 
    print('--------------------- Training GP ---------------------')

    gp.train_model(state, end_train = end_train_date)

    # train lstm model
    print('--------------------- Training LSTM ---------------------')

    df_ = pd.read_csv(f'data/dengue_{state}.csv.gz')

    feat = df_.shape[1]-1
    HIDDEN = 64
    LOOK_BACK = 4
    PREDICT_N = 3

    model = lstm.build_lstm(hidden=HIDDEN, features=feat, predict_n=PREDICT_N, look_back=LOOK_BACK,
                            batch_size=4, loss='mse')


    model.compile(loss='mse', optimizer='adam', metrics=["accuracy", "mape", "mse"])
        
    lstm.train_model(model, state, doenca='dengue',
                    end_train_date=None,
                    ratio = 1,
                    ini_date = '2015-01-01',
                    end_date = end_train_date,
                    filename=f'data/dengue_{state}.csv.gz',
                    min_delta=0.001, label='state',
                    patience = 30, 
                    epochs=300,
                    batch_size=4,
                    predict_n=PREDICT_N,
                    look_back=LOOK_BACK)
