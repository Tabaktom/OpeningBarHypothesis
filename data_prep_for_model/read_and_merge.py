import pandas as pd

symbols = ['AGG', 'DIA', 'IAU', 'IWB', 'QQQ', 'SHY', 'TLT', 'VNQ', 'XLU', 'SPY', 'EEM', 'GDX', 'IEMG', 'IJR', 'USO', 'VWO', 'XLF', 'XLK']
path = '/Users/macbooik/PycharmProjects/OpeningBarHypothesis/data_features_extracted_post2/'
for ind, sym in enumerate(symbols):
    path_full='/Users/macbooik/PycharmProjects/OpeningBarHypothesis/data_features_extracted_post2/simple_features_{}.csv'.format(sym)
    if ind ==0:
        df = pd.read_csv(path_full)
    else:
        n_df = pd.read_csv(path_full)
        df = pd.concat([df, n_df], axis=0)
df=df.reset_index()
df=df.drop(columns=['index'])
df=df.dropna()
df.to_csv('/Users/macbooik/PycharmProjects/OpeningBarHypothesis/Step3_model_fit/ten_ticker.csv')