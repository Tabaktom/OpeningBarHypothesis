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
from keras.callbacks import TensorBoard
import time
NAME = "OpeningBar-5_Submodels-3_HiddenLayersMain_{}".format(time.time())
tensorboard = TensorBoard(log_dir = '/Users/Tom/PycharmProjects/OpeningBarHypothesis/logs/()'.format(NAME))

def board_log():
    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

df = pd.read_csv('ten_ticker.csv')
df=shuffle(df)
df = df.drop(columns=['Date', 'Unnamed: 0'])
print(df)
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
def model_plot(model):
    from keras.utils import plot_model
    plot_model(model, show_shapes = True, show_layer_names = True, rankdir ='TB')



def snn():
    model=tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(360, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def nn(dfs, ms, dows, symbs):
    input1=keras.layers.Input((dfs[1],), dtype='int32')
    mon_model = keras.layers.Dense(1, activation='sigmoid')(input1)

    input2 = keras.layers.Input((ms[1],), type='int32')
    ms_model=keras.layers.Dense(1, activation='sigmoid')(input2)

    input3 = keras.layers.Input((dows[1],), dtype='int32')
    dows_model=keras.layers.Dense(1, activation='sigmoid')(input3)

    input4 = keras.layers.Input((symbs[1],), dtype='int32')
    symbs_model = keras.layers.Dense(1, activation='sigmoid')(input4)

    return

def IndividualFeatureModel(dfs, ms, dows, symbs, prev):

    rest_input=keras.layers.Input((dfs[1],), dtype='float32', name='RestInput')
    rest_model1=keras.layers.Dense(1000, activation='relu',name='RestModel1', kernel_initializer='glorot_uniform')(rest_input)
    rest_model2=keras.layers.Dense(100, activation='relu', name='RestModel2', kernel_initializer='glorot_uniform')(rest_model1)
    rest_model3=keras.layers.Dense(10, activation='relu', name ='Restlayer3', kernel_initializer='glorot_uniform')(rest_model2)
    rest_prob=keras.layers.Dense(4, activation='softmax', name='RestProb', kernel_initializer='glorot_uniform')(rest_model3)

    month_input=keras.layers.Input((ms[1],), dtype='float32', name='MonthInput')
    month_model1=keras.layers.Dense(1000, activation='relu', name='MonthModel1', kernel_initializer='glorot_uniform')(month_input)
    month_model2=keras.layers.Dense(100, activation='relu', name='MonthModel2', kernel_initializer='glorot_uniform')(month_model1)
    month_model3=keras.layers.Dense(10, activation='relu', name = 'MonthModel3', kernel_initializer='glorot_uniform')(month_model2)
    month_prob=keras.layers.Dense(4, activation='softmax', name='MonthProb', kernel_initializer='glorot_uniform')(month_model3)

    dow_input=keras.layers.Input((dows[1],), dtype='float32', name='DowInput')
    dow_model1=keras.layers.Dense(1000, activation='relu', name='DowModel1', kernel_initializer='glorot_uniform')(dow_input)
    dow_model2=keras.layers.Dense(100, activation='relu', name='DowModel2', kernel_initializer='glorot_uniform')(dow_model1)
    dow_model3=keras.layers.Dense(10, activation='relu', name='DowModel3', kernel_initializer='glorot_uniform')(dow_model2)
    dow_prob=keras.layers.Dense(4, activation='softmax', name='Dowprob', kernel_initializer='glorot_uniform')(dow_model3)

    sym_input = keras.layers.Input((symbs[1],), dtype='float32', name='SymbInput')
    sym_model1 = keras.layers.Dense(1000, activation='relu', name='SymbModel1', kernel_initializer='glorot_uniform')(sym_input)
    sym_model2=keras.layers.Dense(100, activation='relu', name='SymbModel2', kernel_initializer='glorot_uniform')(sym_model1)
    sym_model3=keras.layers.Dense(10, activation='relu', name='SymbModel3', kernel_initializer='glorot_uniform')(sym_model2)
    sym_prob = keras.layers.Dense(4, activation='softmax', name='SymbProb', kernel_initializer='glorot_uniform')(sym_model3)

    prev_input=keras.layers.Input((prev[1], ), dtype='float32', name='PrevInput')
    prev_model1=keras.layers.Dense(1000,activation='relu',name='PrevModel1', kernel_initializer='glorot_uniform')(prev_input)
    prev_model2=keras.layers.Dense(100,activation='relu', name='PrevModel2', kernel_initializer='glorot_uniform')(prev_model1)
    prev_model3=keras.layers.Dense(10, activation='relu', name='PrevModel3', kernel_initializer='glorot_uniform')(prev_model2)
    prev_prob=keras.layers.Dense(4, activation='softmax', name='PrevProb', kernel_initializer='glorot_uniform')(prev_model3)


    combined = keras.layers.concatenate([rest_prob, month_prob, dow_prob, sym_prob, prev_prob],name='Combine')
    hidden_1 = keras.layers.Dense(10000, activation='relu', name='FirstHidden', kernel_initializer='glorot_uniform')(combined)
    hidden_2 = keras.layers.Dense(1000, activation='relu', name='SecondHidden', kernel_initializer='glorot_uniform')(hidden_1)
    hidden_3 = keras.layers.Dense(100, activation='relu', name='FinalHidden', kernel_initializer='glorot_uniform')(hidden_2)
    output = keras.layers.Dense(4, activation='softmax', name='Output', kernel_initializer='glorot_uniform')(hidden_3)
    model = keras.Model(inputs=[rest_input, month_input, dow_input, sym_input, prev_input], output=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

###model = functional(dfs, ms, dows, symbs)
###model.fit(train_inputs, train_y, epochs=10)
###res = model.evaluate(test_inputs, test_y)
model = IndividualFeatureModel(dfs, ms, dows, symbs, prev)


history = model.fit(train_inputs,train_y, validation_split=0.2, epochs=70, batch_size=2500, verbose=2, workers=3, use_multiprocessing=True)
results= model.evaluate(test_inputs, test_y)
print('results: {}'.format(results))

#hist = history.history
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
#plotting_accuracy(hist)
