import pandas as pd

symbols = ['EEM', 'GDX', 'IEMG', 'IJR', 'USO', 'VWO', 'XLF', 'XLK', 'IVV']

for sym in symbols:

    daily = pd.read_csv('/Users/Tom/PycharmProjects/OpeningBarHypothesis/local_data/{}_daily.txt'.format(sym))
    hourly = pd.read_csv('/Users/Tom/PycharmProjects/OpeningBarHypothesis/local_data/{}_hourly.txt'.format(sym))

    hourly_frame = {'Date': [], 'hourly_open':[], 'hourly_high': [], 'hourly_low': [], 'hourly_close': [], 'hourly_vol': []}
    for index, row in hourly.iterrows():
        if row.Time == '15:30':
            hourly_frame['Date'].append(str(row.Date)[6:]+'-'+str(row.Date)[:2]+'-'+str(row.Date[3:5]))
            hourly_frame['hourly_open'].append(row.Open)
            hourly_frame['hourly_high'].append(row.High)
            hourly_frame['hourly_low'].append(row.Low)
            hourly_frame['hourly_close'].append(row.Close)
            hourly_frame['hourly_vol'].append(row.Up+row.Down)

    daily_frame = {'Date1': [], 'daily_open': [], 'daily_high': [], 'daily_low': [], 'daily_close':[], 'daily_vol': [], 'symbol': []}

    #print(daily.head(5))
    #print(hourly.head(5))
    #print(daily.tail(5))
    #print(hourly.tail(5))

    for index, row in daily.iterrows():
        if str(row.Date)[6:]+'-'+str(row.Date)[:2]+'-'+str(row.Date[3:5]) in hourly_frame['Date']:
            daily_frame['Date1'].append(str(row.Date)[6:]+'-'+str(row.Date)[:2]+'-'+str(row.Date[3:5]))
            daily_frame['daily_open'].append(row.Open)
            daily_frame['daily_high'].append(row.High)
            daily_frame['daily_low'].append(row.Low)
            daily_frame['daily_close'].append(row.Close)
            daily_frame['daily_vol'].append(row.Vol)
            daily_frame['symbol'].append(sym)



    df = pd.concat([pd.DataFrame(hourly_frame), pd.DataFrame(daily_frame)], axis =1)
    #print(pd.DataFrame(daily_frame))
    #print(pd.DataFrame(hourly_frame))
    print(df)
    frame = {'Label' : [], 'previous_day': []}
    df = df.drop(columns = ['Date1'])
    yesterday = 'NONE'
    for index, row in df.iterrows():
        frame['previous_day'].append(yesterday)
        if row.daily_high == row.hourly_high and row.daily_low == row.hourly_low:
            frame['Label'].append('BOTH') #BOTH
            yesterday='BOTH'
        elif row.daily_high == row.hourly_high:
            frame['Label'].append('HIGHS') #HIGHS
            yesterday='HIGHS'
        elif row.daily_low == row.hourly_low:
            frame['Label'].append('LOWS') #LOWS
            yesterday='LOWS'
        else:
            frame['Label'].append('NONE') #NONE
            yesterday='NONE'
    df = pd.concat([df, pd.DataFrame(frame)], axis=1)
    print(df)
    #df.to_csv('Merged_bar_data.csv')
    print('ok')
    df.to_csv('/Users/Tom/PycharmProjects/OpeningBarHypothesis/merged_data/{}_data.txt'.format(sym))
    print('completed: {}'.format(sym))
