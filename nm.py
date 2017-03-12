import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "theano"
from keras.layers import Dense, LSTM, Convolution1D, Input,Dropout,Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
#from theano.tensor import


def get_data(data_number):
    data = pd.read_csv('./dataset/'+ str(data_number)+'.txt',names=['column_'+str(i) for i in range(18)], delimiter=',')
    data.head()
    X = data.iloc[:,:12].values
    Y = data.iloc[1, -6:].values
    return X,Y

for data_number in range(1,92):
    X,Y = get_data(data_number)

#split data to train test validation


n_dims = 12 #dimension of vector, datas from accelerometr&goroscopes from 2 phones
gest_len = 1182 # how much time steps per gesture
safe_size = 1
num_commands = 6
BATCH_SIZE = 15

opt = Adam(lr=1e-4)

X = np.reshape(X,[1,gest_len,n_dims])
Y = np.reshape(Y,[1,6])
train_data = (X,Y)

nb_epoches = 10

#datas = np.loadtxt("data.txt")
def build_model():
    inp = Input(shape=(gest_len,n_dims))
    d = Convolution1D(nb_filter=10, filter_length=7, input_dim=(n_dims,gest_len), activation="tanh")(inp)
    d = Dropout(0.25)(d)
    #d = Flatten()(d)
    d = LSTM(60,activation="relu")(d)
    out = Dense(6, activation="sigmoid")(d)

    print("model_build")
    return Model(inp,out)


coolNet = build_model()
coolNet.summary()
coolNet.compile(opt,loss="categorical_crossentropy",metrics=['accuracy'])

#def build_mini_batch(datas, batch_size):
#    indices = random.sample(range(0,(datas[0]).shape[1]),batch_size)
#    return (X[:,indices], Y)




#def train_net(nb_epoches, model, safe_steps, data):
#    for n in range(nb_epoches):
#        dts = build_mini_batch(data,BATCH_SIZE)
#        loss = model.train_on_batch(dts[0],dts[1])
#        if(nb_epoches % safe_size ==0):
#            fpth = "model-{n}.h5"
#            model.save(fpth)
#            losses.append(loss)




    #model.fit(train_data[0],train_data[1],batch_size=BATCH_SIZE, nb_epoch = safe_steps)
    #filepath="model-{epoch:02d}-{loss:.4f}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    #return callbacks_list

#coolNet.save("model.h5")

#def training():

def training(nb_epo, safe_step):
    for x in range(nb_epo):
        coolNet.fit(train_data[0],train_data[1],nb_epoch=safe_step, batch_size=BATCH_SIZE)
        if(nb_epo // safe_step == 0):
            coolNet.save('%s.h5' % x)


#print(type(X))
#print(np.shape(X))
#coolNet.fit(train_data[0],train_data[1], nb_epoch=10,batch_size=5)
training(10,5)
#coolNet.save("model.h5")
#train_net(10,coolNet,2, train_data)
#coolNet.predict(X)