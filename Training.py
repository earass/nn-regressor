import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from sklearn.metrics import mean_squared_error

def get_training_data():
    # load training examples
    Xtr = data_loader("TrainData.csv")
    print('loaded training examples')

    # load training labels
    Ytr = data_loader("TrainLabels.csv")
    print('loaded training labels')

    return Xtr, Ytr

def data_loader(filename):
    data = np.loadtxt(filename)
    return data

def baseline_model(lr=0.001,hidden_layer_neurons=1,dropout_rate=0.2):
    """ defining the model """
    # create model
    model = Sequential()

    # input layer
    model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))

    # hidden layer
    model.add(Dense(hidden_layer_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(1, kernel_initializer='normal'))

    # create adam optimizer object
    adam = optimizers.Adam(lr=lr)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

def train(X,y):
    best_params = {'batch_size': 5, 'epochs': 100, 'lr': 0.001}
    batch_size = best_params['batch_size']
    lr = best_params['lr']
    epochs = best_params['epochs']
    model = baseline_model(lr=lr)
    model.fit(X,y,batch_size=batch_size,epochs=epochs,verbose=1)
    return model

def execute_training(save_model=False):
    X, y = get_training_data()
    trained_model = train(X, y)
    if save_model:
        trained_model.save('myModel.h5')
    return trained_model

if __name__ == '__main__':
    execute_training()

