import h5py
import numpy as np
from keras.utils import np_utils

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy import genfromtxt

input_file = "full_dataset_vectors.h5"
h5file = h5py.File(input_file,"r")
X_train = h5file["X_train"][:]
y_train = h5file["y_train"][:]
X_test = h5file["X_test"][:]  
y_test = h5file["y_test"][:]
"""
img_width, img_height, img_depth = 16, 16, 16
img_channels = 1
nb_classes = 10
X_train = X_train.reshape(X_train.shape[0], img_depth, img_height, img_width, img_channels)
X_test = X_test.reshape(X_test.shape[0], img_depth, img_height, img_width, img_channels)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
"""
img_width, img_height, img_depth = 16, 16, 16
deta=X_train[0]
deta = deta.reshape(img_depth, img_height, img_width)

# グラフ作成
fig = pyplot.figure()
ax = Axes3D(fig)

# 軸ラベルの設定
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# 表示範囲の設定
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
#"""
# グラフ描画
ax.plot(deta)
pyplot.show()
#"""