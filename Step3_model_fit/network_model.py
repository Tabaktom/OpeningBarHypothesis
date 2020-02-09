import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
import tensorflow as tf
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.models import Model
import keras


def board_log():
    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

df = pd.read_csv('ten_ticker.csv')
df=shuffle(df)
print(len(df))
df = df.drop(columns=['Date', 'Unnamed: 0'])
#print(df.columns)

labels = df.pop('Label')
label_set = set(labels.values)
label_set_ids = {lab: id for id, lab in enumerate(label_set)}
final_labels = [label_set_ids[lab] for lab in labels]

tickers=df.pop('symbol')
ticker_set=set(tickers.values)
ticker_set_ids={tick:id for id,tick in enumerate(ticker_set)}
final_ticker=[ticker_set_ids[tick] for tick in tickers]
df['symbol'] = final_ticker

month=df.pop('Month')
dayofweek=df.pop('DayofWeek')
symbol=df.pop('symbol')

month=to_categorical(month.values)
dayofweek=to_categorical(dayofweek.values)
symbol=to_categorical(symbol.values)
#print(symbol)

dfs= df.shape
ms= month.shape
dows = dayofweek.shape
symbs= symbol.shape
df=np.concatenate((df.values, month, dayofweek, symbol), axis=1)

print(type(ms[1]))
print('df shape: {}'.format(df.shape))
print('dfs: {}'.format(dfs))
print('ms: {}'.format(ms))
print('dows: {}'.format(dows))
print('symbs: {}'.format(symbs))

#print(df)
x = df
y= np.array(final_labels)
#print(y)

train_x=x[:int(len(x)*0.7)]
train_y=y[:int(len(y)*0.7)]
test_x=x[int(len(x)*0.7):]
test_y=y[int(len(y)*0.7):]

first = dfs[1]
second = dfs[1]+ms[1]
third = dfs[1] +ms[1]+dows[1]
fourth = dfs[1]+ms[1]+dows[1]+symbs[1]
print(first, second, third, fourth)

train_inputs = [train_x[:, :first], train_x[:, first:second], train_x[: ,second:third],
                train_x[ :, third:fourth]]
test_inputs = [test_x[:, :first], test_x[:, first:second], test_x[: ,second:third],
               test_x[ :, third:fourth]]

#print(test_x.shape)
#data = tf.data.Dataset.from_tensor_slices((x, y))

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

def nn(mon, dow,sym, df, lab):

    #define model
    #embedded layer1- input month cols
    inp1=keras.layers.Input((mon.shape[0],))
    layer1 = Dense(40, activation='relu')(mon)
    #classifier layer 1 - softmax
    output_layer1 = Dense(4, activation='softmax')(layer1)
    #embedded layer 2 - input DoW cols
    layer2 = Dense(dow.shape[1], activation='relu')(dow)
    #classifier layer 2 - softmax
    output_layer2= Dense(4, activation='softmax')(layer2)
    #embedded layer 3- input symbol cols
    layer3= Dense(sym.shape[1], activation='relu')(sym)
    #classifier layer 3 - softmax
    output_layer3=Dense(4, activation='relu')(layer3)
    #embedded layer 4 - rest cols
    layer4=Dense(df.shape[1], activation='relu')(df)
    #classifier layer 4 -softmax
    output_layer4 = Dense(4, activation='softmax')(layer4)
    #concatenate outputs of all classifier layers

    embedded_output = keras.layers.Concatenate(axis=-1)([np.asarray(output_layer1), np.asarray(output_layer2),
                                                         np.asarray(output_layer3), np.asarray(output_layer4)])

    #define a sequential model input layer, dimensions: 16 (4 classes x 4 models feeding in)
    BigModel = Sequential()
    # some hidden layers - relu
    BigModel.add(Dense(embedded_output.shape[1], activation ='relu'))
    BigModel.add(Dense(embedded_output.shape[1]*8, activation='relu'))
    BigModel.add(Dense(embedded_output.shape[1]*32, activation='relu'))
    BigModel.add(Dense(embedded_output.shape[1]*64, activation='relu'))
    BigModel.add(Dense(embedded_output.shape[1]*32, activation='relu'))
    BigModel.add(Dense(embedded_output.shape[1]*8, activation='relu'))
    BigModel.add(Dense(embedded_output[1], activation='relu'))
    #output layer -softmax
    BigModel.add(Dense(4, activation='softmax'))
    return

def functional(dfs, ms, dows, symbs):

    rest_input = keras.layers.Input((dfs[1],), dtype='float32')
    #rest_embedding = keras.layers.embeddings(10,)(rest_input)
    rest_prob= keras.layers,Dense(4, activation='softmax')(rest_input)

    month_input = keras.layers.Input((ms[1],), dtype='float32')
    #month_embedding = keras.layers.embeddings(11, )(month_input)
    month_prob = keras.layers.Dense(4, activation='softmax')(month_input)

    dow_input = keras.layers.Input((dows[1],), dtype='float32')
    #dow_embedding = keras.layers.embeddings(5,)(dow_input)
    dow_prob = keras.layers.Dense(4, activation='softmax')(dow_input)

    sym_input = keras.layers.Input((symbs[1],), dtype='float32')
    #sym_embedding = keras.layers.embeddings(18,)(sym_input)
    sym_prob = keras.layers.Dense(4, activation='softmax')(sym_input)



    combined = keras.layers.Dot(axes =-1, normalize=False)([rest_prob, month_prob, dow_prob, sym_prob])
    hidden_1 = keras.layers.Dense(64, activation='relu')(combined)
    hidden_2 = keras.layers.Dense(128, activation='relu')(hidden_1)
    hidden_3 = keras.layers.Dense(64, activation='relu')(hidden_2)
    output = keras.layers.Dense(4, activation='softmax')(hidden_3)

    model = keras.Model(inputs=[rest_input, month_input, dow_input, sym_input], output=[output])
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop')
    return model

###model = functional(dfs, ms, dows, symbs)
###model.fit(train_inputs, train_y, epochs=10)
###res = model.evaluate(test_inputs, test_y)
model = snn()
model.fit(train_x,train_y, epochs=20)
results= model.evaluate(test_x, test_y)
print('results: {}'.format(results))
board_log()
#hist = history.history
def plotting_loss(history):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)
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
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')  # "bo" is for "blue dot"
    plt.plot(epochs, val_acc, 'b',
             label='Validation accuracy')  # b is for "solid blue line" >>> plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

#plotting_accuracy(hist)
