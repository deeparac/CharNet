from mxnet import gluon

def char_cnn():
    "See Zhang and LeCun, 2015"
    
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv1D(256, 7, activation='relu'))
        net.add(gluon.nn.MaxPool1D(3))
        net.add(gluon.nn.Conv1D(256, 7, activation='relu'))
        net.add(gluon.nn.MaxPool1D(3))
        net.add(gluon.nn.Conv1D(256, 3, activation='relu'))
        net.add(gluon.nn.Conv1D(256, 3, activation='relu'))
        net.add(gluon.nn.Conv1D(256, 3, activation='relu'))
        net.add(gluon.nn.Conv1D(256, 3, activation='relu'))
        net.add(gluon.nn.MaxPool1D(3))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(1024, activation="relu"))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(1024, activation="relu"))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(1))
        
    return net

model = char_cnn()