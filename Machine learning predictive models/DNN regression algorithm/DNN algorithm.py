import xlrd
import numpy as np
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization

def loadExcel(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # Line numbers
    ncols = table.ncols  # Column numbers
    datamatrix = np.zeros((nrows-1, ncols))
    for i in range(1, nrows):
        rows = table.row_values(i)
        datamatrix[i-1, :] = rows
    return datamatrix[:,:ncols-1], datamatrix[:,-1]

def do_MinMaxScaler(data):
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    result = scaler.transform(data)
    return result

def do_PCA(data, feature_num):
    pca = PCA(n_components=feature_num)
    pca.fit(data)
    return pca.transform(data)

def getModel():
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})

    return svr

def do_preprocess(data):
    new_data = do_MinMaxScaler(data)
    new_data = do_PCA(new_data, 6)
    return new_data



pathX = 'DNN.xls'  #
attrs, output = loadExcel(pathX)
attrs = do_preprocess(attrs)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(attrs, output, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

model = Sequential()
model.add(Conv1D(8, 3, input_shape=(X_train.shape[1],1), activation='relu', padding='same'))
# model.add(Conv1D(8, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Dropout(rate=0.2))

model.add(Conv1D(16, 3, activation='relu', padding='same'))
# model.add(Conv1D(16, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(2, padding="same"))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='linear'))

# optimizer = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
optimizer = optimizers.Adam(lr=0.001);
model.compile(optimizer=optimizer,
              loss='mae',
              metrics=['mae'])
hist = model.fit(X_train, Y_train,
          batch_size=64, epochs=2000,
          validation_data=(X_val, Y_val))
res = model.evaluate(X_test, Y_test)[1]
print(res)

model.fit(attrs, output,
          epochs=2000, batch_size=16, verbose=1)

predict = model.predict(attrs)
predict
import pandas as pd
import numpy as np
predict = pd.DataFrame(predict)
outputpath='C:/Users/90630/PycharmProjects/pythonProject1/DNN/230-DNN-predicted.csv'
predict.to_csv(outputpath,sep=',',index=False,header=True)



plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

