import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
import tensorflow as tf
from sklearn.utils import shuffle
from keras.utils import to_categorical, plot_model
from keras.models import Model
import keras
import time
import model_functions
from keras.utils import plot_model
from keras.utils import vis_utils


df = pd.read_csv('ten_ticker.csv')
df=shuffle(df)
df = df.drop(columns=['Date', 'Unnamed: 0'])

labels = df.pop('Label')
label_set = set(labels.values)
label_set_ids = {lab: id for id, lab in enumerate(label_set)}
final_labels = [label_set_ids[lab] for lab in labels]

tickers=df.pop('symbol')
ticker_set=set(tickers.values)
ticker_set_ids={tick:id for id,tick in enumerate(ticker_set)}
final_ticker=[ticker_set_ids[tick] for tick in tickers]
df['symbol'] = final_ticker

yesterday = df.pop('previous_day')
yesterday_set=set(yesterday.values)
yesterday_set_ids = {yes:id for id, yes in enumerate(yesterday_set)}
final_yesterday = [yesterday_set_ids[yes] for yes in yesterday]
df['previous_day'] = final_yesterday

month=df.pop('Month')
dayofweek=df.pop('DayofWeek')
symbol=df.pop('symbol')
previous_day = df.pop('previous_day')

month=to_categorical(month.values)
dayofweek=to_categorical(dayofweek.values)
symbol=to_categorical(symbol.values)
previous_day=to_categorical(previous_day.values)

dfs, ms, dows, symbs, prev= df.shape, month.shape, dayofweek.shape, symbol.shape, previous_day.shape
df=np.concatenate((df.values, month, dayofweek, symbol, previous_day), axis=1)
x, y = df, np.array(final_labels)
train_test = int(len(x)*0.9)
train_x, train_y=x[:train_test], y[:train_test]
test_x, test_y=x[train_test:], y[train_test:]
first, second, third, fourth, fifth = dfs[1], dfs[1]+ms[1], dfs[1] +ms[1]+dows[1], dfs[1]+ms[1]+dows[1]+symbs[1], dfs[1]+ms[1]+dows[1]+symbs[1]+prev[1]

train_val_split=int(0.9*len(train_x))
train_inputs = [train_x[:, :first], train_x[:, first:second], train_x[: ,second:third],
                train_x[:, third:fourth], train_x[:, fourth:fifth]]
test_inputs = [test_x[:, :first], test_x[:, first:second], test_x[: ,second:third],
               test_x[:, third:fourth], test_x[:, fourth:fifth]]




#model = model_functions.FiveIntoOne(dfs, ms, dows, symbs, prev)

model = model_functions.FiveIntoTwoIntoOne(dfs, ms, dows, symbs, prev)
#plot_model(model, to_file='five_into_two_into_one_model.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=60)
#imageio.imwrite(image, 'model.JPG')

history = model.fit(train_inputs,train_y, validation_split=0.2, epochs=100, batch_size=250, verbose=2, workers=3, use_multiprocessing=True)
results= model.evaluate(test_inputs, test_y)
print('results: {}'.format(results))

def plotting_loss(history):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    history_dict = history.history
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')  # "bo" is for "blue dot"
    plt.plot(epochs, val_loss, 'b',
             label='Validation loss')  # b is for "solid blue line" >>> plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plotting_accuracy(history):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')  # "bo" is for "blue dot"
    plt.plot(epochs, val_acc, 'b',
             label='Validation accuracy')  # b is for "solid blue line" >>> plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plotting_accuracy(history)
plotting_loss(history)
plotting_accuracy(history)
