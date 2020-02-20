import keras
import tensorflow as tf

def FiveIntoTwoIntoOne(dfs, ms, dows, symbs, prev):

    rest_input=keras.layers.Input((dfs[1],), dtype='float32', name='RestInput')
    rest_model1=keras.layers.Dense(16, activation='relu',name='RestModel1', kernel_initializer='glorot_uniform')(rest_input)
    rest_model2=keras.layers.Dense(64, activation='relu', name='RestModel2', kernel_initializer='glorot_uniform')(rest_model1)
    rest_model3 = keras.layers.Dense(128, activation='relu', name='RestModel3', kernel_initializer='glorot_uniform')(rest_model2)
    rest_model4 = keras.layers.Dense(64, activation='relu', name='RestModel4', kernel_initializer='glorot_uniform')(rest_model3)
    rest_model5=keras.layers.Dense(16, activation='relu', name ='Restlayer5', kernel_initializer='glorot_uniform')(rest_model4)
    rest_prob=keras.layers.Dense(4, activation='softmax', name='RestProb', kernel_initializer='glorot_uniform')(rest_model5)

    month_input=keras.layers.Input((ms[1],), dtype='float32', name='MonthInput')
    month_model1=keras.layers.Dense(16, activation='relu', name='MonthModel1', kernel_initializer='glorot_uniform')(month_input)
    month_model2=keras.layers.Dense(64, activation='relu', name='MonthModel2', kernel_initializer='glorot_uniform')(month_model1)
    month_model3=keras.layers.Dense(128, activation='relu', name = 'MonthModel3', kernel_initializer='glorot_uniform')(month_model2)
    month_model4 = keras.layers.Dense(64, activation='relu', name='MonthModel4', kernel_initializer='glorot_uniform')(month_model3)
    month_model5 = keras.layers.Dense(16, activation='relu', name='MonthModel5', kernel_initializer='glorot_uniform')(month_model4)
    month_prob=keras.layers.Dense(4, activation='softmax', name='MonthProb', kernel_initializer='glorot_uniform')(month_model5)

    dow_input=keras.layers.Input((dows[1],), dtype='float32', name='DowInput')
    dow_model1=keras.layers.Dense(16, activation='relu', name='DowModel1', kernel_initializer='glorot_uniform')(dow_input)
    dow_model2=keras.layers.Dense(64, activation='relu', name='DowModel2', kernel_initializer='glorot_uniform')(dow_model1)
    dow_model3=keras.layers.Dense(128, activation='relu', name='DowModel3', kernel_initializer='glorot_uniform')(dow_model2)
    dow_model4 = keras.layers.Dense(64, activation='relu', name='DowModel4', kernel_initializer='glorot_uniform')(dow_model3)
    dow_model5 = keras.layers.Dense(16, activation='relu', name='DowModel5', kernel_initializer='glorot_uniform')(dow_model4)
    dow_prob=keras.layers.Dense(4, activation='softmax', name='Dowprob', kernel_initializer='glorot_uniform')(dow_model5)

    sym_input = keras.layers.Input((symbs[1],), dtype='float32', name='SymbInput')
    sym_model1 = keras.layers.Dense(16, activation='relu', name='SymbModel1', kernel_initializer='glorot_uniform')(sym_input)
    sym_model2=keras.layers.Dense(64, activation='relu', name='SymbModel2', kernel_initializer='glorot_uniform')(sym_model1)
    sym_model3=keras.layers.Dense(128, activation='relu', name='SymbModel3', kernel_initializer='glorot_uniform')(sym_model2)
    sym_model4 = keras.layers.Dense(64, activation='relu', name='SymbModel4', kernel_initializer='glorot_uniform')(sym_model3)
    sym_model5 = keras.layers.Dense(16, activation='relu', name='SymbModel5', kernel_initializer='glorot_uniform')(sym_model4)
    sym_prob = keras.layers.Dense(4, activation='softmax', name='SymbProb', kernel_initializer='glorot_uniform')(sym_model5)

    prev_input=keras.layers.Input((prev[1], ), dtype='float32', name='PrevInput')
    prev_model1=keras.layers.Dense(16,activation='relu',name='PrevModel1', kernel_initializer='glorot_uniform')(prev_input)
    prev_model2=keras.layers.Dense(64,activation='relu', name='PrevModel2', kernel_initializer='glorot_uniform')(prev_model1)
    prev_model3=keras.layers.Dense(128, activation='relu', name='PrevModel3', kernel_initializer='glorot_uniform')(prev_model2)
    prev_model4 = keras.layers.Dense(64, activation='relu', name='PrevModel4', kernel_initializer='glorot_uniform')(prev_model3)
    prev_model5 = keras.layers.Dense(16, activation='relu', name='PrevModel5', kernel_initializer='glorot_uniform')(prev_model4)
    prev_prob=keras.layers.Dense(4, activation='softmax', name='PrevProb', kernel_initializer='glorot_uniform')(prev_model5)

    dow_month_symb = keras.layers.concatenate([dow_prob, month_prob, sym_prob],name='dow_month_symb')
    prev_rest = keras.layers.concatenate([prev_prob, rest_prob], name = 'prev_rest')

    dow_month_symb_hidden1 = keras.layers.Dense(12, activation='relu', name = 'DMS_hidden1', kernel_initializer='glorot_uniform')(dow_month_symb)
    dow_month_symb_hidden2 = keras.layers.Dense(48, activation='relu', name = 'DMS_hidden2', kernel_initializer='glorot_uniform')(dow_month_symb_hidden1)
    dow_month_symb_hidden3 = keras.layers.Dense(192, activation='relu', name ='DMS_hidden3', kernel_initializer = 'glorot_uniform')(dow_month_symb_hidden2)
    dow_month_symb_hidden4 = keras.layers.Dense(48, activation='relu', name='DMS_hidden4', kernel_initializer='glorot_uniform')(dow_month_symb_hidden3)
    dow_month_symb_hidden5=keras.layers.Dense(12, activation='relu', name='DMS_hidden5', kernel_initializer='glorot_uniform')(dow_month_symb_hidden4)
    dow_month_symb_prob = keras.layers.Dense(4, activation='softmax', name ='DMS_prob', kernel_initializer='glorot_uniform')(dow_month_symb_hidden5)

    prev_rest_hidden1 = keras.layers.Dense(8, activation='relu', name = 'PR_hidden1', kernel_initializer='glorot_uniform')(prev_rest)
    prev_rest_hidden2 = keras.layers.Dense(32, activation='relu', name = 'PR_hidden2', kernel_initializer='glorot_uniform')(prev_rest_hidden1)
    prev_rest_hidden3 = keras.layers.Dense(128, activation='relu', name ='PR_hidden3', kernel_initializer = 'glorot_uniform')(prev_rest_hidden2)
    prev_rest_hidden4 = keras.layers.Dense(32, activation='relu', name='PR_hidden4', kernel_initializer='glorot_uniform')(prev_rest_hidden3)
    prev_rest_hidden5=keras.layers.Dense(8, activation='relu', name='PR_hidden5', kernel_initializer='glorot_uniform')(prev_rest_hidden4)
    prev_rest_prob = keras.layers.Dense(4, activation='softmax', name ='PR_prob', kernel_initializer='glorot_uniform')(prev_rest_hidden5)

    combined = keras.layers.concatenate([dow_month_symb_prob, prev_rest_prob], name = 'combined')
    hidden_1 = keras.layers.Dense(8, activation='relu', name='FinalHidden1', kernel_initializer='glorot_uniform')(combined)
    hidden_2 = keras.layers.Dense(32, activation='relu', name='FinalHidden2', kernel_initializer='glorot_uniform')(hidden_1)
    hidden_3 = keras.layers.Dense(128, activation='relu', name='FinalHidden3', kernel_initializer='glorot_uniform')(hidden_2)
    hidden_4 = keras.layers.Dense(32, activation='relu', name='FinalHidden4', kernel_initializer='glorot_uniform')(hidden_3)
    hidden_5 = keras.layers.Dense(8, activation='relu', name='FinalHidden5', kernel_initializer='glorot_uniform')(hidden_4)
    output = keras.layers.Dense(4, activation='softmax', name='Output', kernel_initializer='glorot_uniform')(hidden_5)
    model = keras.Model(inputs=[rest_input, month_input, dow_input, sym_input, prev_input], output=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def FiveIntoOne(dfs, ms, dows, symbs, prev):

    rest_input=keras.layers.Input((dfs[1],), dtype='float32', name='RestInput')
    rest_model1=keras.layers.Dense(16, activation='relu',name='RestModel1', kernel_initializer='glorot_uniform')(rest_input)
    rest_model2=keras.layers.Dense(64, activation='relu', name='RestModel2', kernel_initializer='glorot_uniform')(rest_model1)
    rest_model3 = keras.layers.Dense(128, activation='relu', name='RestModel3', kernel_initializer='glorot_uniform')(rest_model2)
    rest_model4 = keras.layers.Dense(64, activation='relu', name='RestModel4', kernel_initializer='glorot_uniform')(rest_model3)
    rest_model5=keras.layers.Dense(16, activation='relu', name ='Restlayer5', kernel_initializer='glorot_uniform')(rest_model4)
    rest_prob=keras.layers.Dense(4, activation='softmax', name='RestProb', kernel_initializer='glorot_uniform')(rest_model5)

    month_input=keras.layers.Input((ms[1],), dtype='float32', name='MonthInput')
    month_model1=keras.layers.Dense(16, activation='relu', name='MonthModel1', kernel_initializer='glorot_uniform')(month_input)
    month_model2=keras.layers.Dense(64, activation='relu', name='MonthModel2', kernel_initializer='glorot_uniform')(month_model1)
    month_model3=keras.layers.Dense(128, activation='relu', name = 'MonthModel3', kernel_initializer='glorot_uniform')(month_model2)
    month_model4 = keras.layers.Dense(64, activation='relu', name='MonthModel4', kernel_initializer='glorot_uniform')(month_model3)
    month_model5 = keras.layers.Dense(16, activation='relu', name='MonthModel5', kernel_initializer='glorot_uniform')(month_model4)
    month_prob=keras.layers.Dense(4, activation='softmax', name='MonthProb', kernel_initializer='glorot_uniform')(month_model5)

    dow_input=keras.layers.Input((dows[1],), dtype='float32', name='DowInput')
    dow_model1=keras.layers.Dense(16, activation='relu', name='DowModel1', kernel_initializer='glorot_uniform')(dow_input)
    dow_model2=keras.layers.Dense(64, activation='relu', name='DowModel2', kernel_initializer='glorot_uniform')(dow_model1)
    dow_model3=keras.layers.Dense(128, activation='relu', name='DowModel3', kernel_initializer='glorot_uniform')(dow_model2)
    dow_model4 = keras.layers.Dense(64, activation='relu', name='DowModel4', kernel_initializer='glorot_uniform')(dow_model3)
    dow_model5 = keras.layers.Dense(16, activation='relu', name='DowModel5', kernel_initializer='glorot_uniform')(dow_model4)
    dow_prob=keras.layers.Dense(4, activation='softmax', name='Dowprob', kernel_initializer='glorot_uniform')(dow_model5)

    sym_input = keras.layers.Input((symbs[1],), dtype='float32', name='SymbInput')
    sym_model1 = keras.layers.Dense(16, activation='relu', name='SymbModel1', kernel_initializer='glorot_uniform')(sym_input)
    sym_model2=keras.layers.Dense(64, activation='relu', name='SymbModel2', kernel_initializer='glorot_uniform')(sym_model1)
    sym_model3=keras.layers.Dense(128, activation='relu', name='SymbModel3', kernel_initializer='glorot_uniform')(sym_model2)
    sym_model4 = keras.layers.Dense(64, activation='relu', name='SymbModel4', kernel_initializer='glorot_uniform')(sym_model3)
    sym_model5 = keras.layers.Dense(16, activation='relu', name='SymbModel5', kernel_initializer='glorot_uniform')(sym_model4)
    sym_prob = keras.layers.Dense(4, activation='softmax', name='SymbProb', kernel_initializer='glorot_uniform')(sym_model5)

    prev_input=keras.layers.Input((prev[1], ), dtype='float32', name='PrevInput')
    prev_model1=keras.layers.Dense(16,activation='relu',name='PrevModel1', kernel_initializer='glorot_uniform')(prev_input)
    prev_model2=keras.layers.Dense(64,activation='relu', name='PrevModel2', kernel_initializer='glorot_uniform')(prev_model1)
    prev_model3=keras.layers.Dense(128, activation='relu', name='PrevModel3', kernel_initializer='glorot_uniform')(prev_model2)
    prev_model4 = keras.layers.Dense(64, activation='relu', name='PrevModel4', kernel_initializer='glorot_uniform')(prev_model3)
    prev_model5 = keras.layers.Dense(16, activation='relu', name='PrevModel5', kernel_initializer='glorot_uniform')(prev_model4)
    prev_prob=keras.layers.Dense(4, activation='softmax', name='PrevProb', kernel_initializer='glorot_uniform')(prev_model5)


    combined = keras.layers.concatenate([rest_prob, month_prob, dow_prob, sym_prob, prev_prob],name='Combine')
    hidden_1 = keras.layers.Dense(16, activation='relu', name='FirstHidden', kernel_initializer='glorot_uniform')(combined)
    hidden_2 = keras.layers.Dense(64, activation='relu', name='SecondHidden', kernel_initializer='glorot_uniform')(hidden_1)
    hidden_3 = keras.layers.Dense(128, activation='relu', name='FinalHidden1', kernel_initializer='glorot_uniform')(hidden_2)
    hidden_4 = keras.layers.Dense(64, activation='relu', name='FinalHidden2', kernel_initializer='glorot_uniform')(hidden_3)
    hidden_5 = keras.layers.Dense(16, activation='relu', name='FinalHidden3', kernel_initializer='glorot_uniform')(hidden_4)


    output = keras.layers.Dense(4, activation='softmax', name='Output', kernel_initializer='glorot_uniform')(hidden_5)
    model = keras.Model(inputs=[rest_input, month_input, dow_input, sym_input, prev_input], output=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def NormalSequential():
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