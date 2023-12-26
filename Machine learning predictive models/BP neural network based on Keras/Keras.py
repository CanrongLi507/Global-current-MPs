import pandas as pd
from dask.optimization import inline

data = pd.read_csv("C:/Users/90630/PycharmProjects/pythonProject1/Keras.csv")
train_data = data.loc[0:229, ['Lon','Lat','Sort','uo','vo','VHM0','VMDR','VSDX','VSDY','THETA','SALT','VVEL','UVEL','Pressure']]
train_data.shape

# Read the data in rows 0 to 163 and columns A to I of the data set #

train_targets = data.loc[0:229, ['abundance']]
train_targets.shape

# Read lines 0 to 163 and abundance of the data set #

test_data = data.loc[230:168721, ['Lon','Lat','Sort','uo','vo','VHM0','VMDR','VSDX','VSDY','THETA','SALT','VVEL','UVEL','Pressure']]

# Data preprocessing #
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# Compile the DNN model #
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GTX 1050 Ti

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(232, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(232, activation='relu'))
    model.add(layers.Dense(232, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Partition verification set, K-fold cross-validation #
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # Preparing verification data: Data for the K partition #
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare training data: Data for all other partitions #
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Building the Keras Model (compiled) #
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=1)
    # Evaluate the model on validation data #
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

# View MAE average #
np.mean(all_scores)

# Training model #
from keras import backend as K
K.clear_session()

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()

    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=1)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# drawing #
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[:100])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

model = build_model()
# Train it on the entirety of the data #
model.fit(train_data, train_targets,
          epochs=600, batch_size=16, verbose=1)

predict = model.predict(test_data)

predict
import pandas as pd
import numpy as np
predict = pd.DataFrame(predict)
outputpath='C:/Users/90630/Desktop/Pathways to global sediment analysis/230-Keras-predictd.csv'
predict.to_csv(outputpath,sep=',',index=False,header=True)



%matplotlib inline
import matplotlib.pylab as plt
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
x = np.arange(1,231)
plt.title("MPs predict")
plt.xlabel("Regional division")
plt.ylabel("MPs abundance")
plt.plot(range(0, 230, 1),predict)
plt.savefig(r"C:\Users\90630\PycharmProjects\pythonProject1\train2-2.jpg", dpi=500, bbox_inches='tight')
plt.show()

# dataframe Derived data #
outputpath='C:/Users/90630/PycharmProjects/pythonProject1/Keras/230-Keras-predictd.csv'
predict.to_csv(outputpath,sep=',',index=False,header=True)