import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import resnet3d

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet101_3d-mnist')

batch_size = 32	#バッチサイズ
nb_classes = 10	#出力データ配列数
nb_epoch = 200	#エポック数
data_augmentation = False
img_width, img_height, img_depth = 16, 16, 16	#横,縦,高さ
img_channels = 1	#チャンネル数

#3d-mnistのデータセットをロード
input_file = "full_dataset_vectors.h5"
h5file = h5py.File(input_file,"r")
X_train = h5file["X_train"][:]
y_train = h5file["y_train"][:]
X_test = h5file["X_test"][:]  
y_test = h5file["y_test"][:]

#数字を配列に変更　例) 7 = [0][0][0][0][0][0][0][1][0][0]
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#(データ数,高さ,縦,横,チャンネル数)の配列に変更
X_train = X_train.reshape(X_train.shape[0], img_depth, img_height, img_width, img_channels)
X_test = X_test.reshape(X_test.shape[0], img_depth, img_height, img_width, img_channels)
#float32に型変換
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#101層の3d-resnetモデルを作成 
model = resnet3d.Resnet3DBuilder.build_resnet_101(
(img_depth, img_height, img_width, img_channels), nb_classes)

# loss, optimizer, metricsを選択
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0.,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
