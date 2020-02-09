import pandas as pd
import numpy as np
import datetime


df = pd.read_csv('data.txt')

date_frame ={'Month': [], 'DayofWeek': [], 'symbol': []}

years = []
months = []
weekdays= []
print(df.head(5))
print(df.columns)
for index, row in df.iterrows():
    #if row.Date[0:4] not in years:
        #years.append(row.Date[0:4])
    if row.Date[5:7] not in months:
        months.append(row.Date[5:7])
    d =datetime.date(int(row.Date[0:4]), int(row.Date[5:7]), int(row.Date[8:]))
    date_frame['DayofWeek'].append(d.weekday())
for index, row in df.iterrows():
    #y = row.Date[0:4]
    #date_frame['Year'].append(years.index(y))
    m = row.Date[5:7]
    date_frame['Month'].append(int(m))
    date_frame['symbol'].append('SPY')

date_frame = (pd.DataFrame(date_frame))
#date_frame = date_frame.drop(columns= ['Year'])
technical_frame = {'vol_percent_prev' :[], 'close_percent_prev': [],
                   'open_percent_prev' : [], 'high_percent_prev' : [],
                   'low_percent_prev': [],'prev_day_movement' : [],
                   'bar_movement': [], 'close_percent_high': [], 'close_percent_low': []}

print(technical_frame.keys())
for index, row in df.iterrows():
    if index == 0:
        for key in technical_frame.keys():
            technical_frame[key].append(0)
    else:
        technical_frame['vol_percent_prev'].append(row.hourly_vol/df.daily_vol[index-1])
        technical_frame['close_percent_prev'].append(row.hourly_close/df.daily_close[index-1])
        technical_frame['open_percent_prev'].append(row.hourly_open/df.daily_open[index-1])
        technical_frame['high_percent_prev'].append(row.hourly_high/df.daily_high[index-1])
        technical_frame['low_percent_prev'].append(row.hourly_low/df.daily_low[index-1])
        technical_frame['prev_day_movement'].append((df.daily_close[index-1]-df.daily_open[index-1])/df.daily_open[index-1])
        technical_frame['bar_movement'].append((row.hourly_close-row.hourly_open) / row.hourly_open)
        technical_frame['close_percent_high'].append(row.hourly_close/row.hourly_high)
        technical_frame['close_percent_low'].append(row.hourly_low/row.hourly_close)


technical_frame['Label'] = df.Label.values.tolist()

dataf = pd.concat([pd.DataFrame(date_frame), pd.DataFrame(technical_frame)], axis =1)
dataf.index=df.Date
print(dataf)
dataf.to_csv('/Users/macbooik/PycharmProjects/OpeningBarHypothesis/data_features_extracted_post2/simple_features_{}.csv'.format('SPY'))

