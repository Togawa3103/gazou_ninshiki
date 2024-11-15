
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%config InlineBackend.figure_formats = {'png', 'retina'}

import os, zipfile, io, re
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception
from keras.models import Model, load_model
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 100
classes = ["male", "female"]
num_classes = len(classes)

dir_path="part1"
#dir_path="test"
#%%time
# ZIP読み込み
imgfiles = os.listdir(dir_path)
#print(z.namelist())
# 画像ファイルパスのみ取得
#imgfiles = [ x for x in z.namelist()]
#print(imgfiles[0])
#exit()
X = []
Y = []
for imgfile in imgfiles:
    #print(imgfile)
    #exit()
    # ZIPから画像読み込み
    image = Image.open(dir_path+"/"+imgfile)
    # RGB変換
    image = image.convert('RGB')
    # リサイズ
    image = image.resize((image_size, image_size))
    # 画像から配列に変換
    data = np.asarray(image)
    file = os.path.basename(imgfile)
    #print(file)
    file_split = [i for i in file.split('_')]
    #print(file_split)
    X.append(data)
    Y.append(float(file_split[0]))
    image.close()
del imgfiles
#print(Y)
#print(Y.count(0),Y.count(1))

X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)
#print(np.count_nonzero(Y == '0'))
#print(np.count_nonzero(Y == '1'))
#print(np.unique(Y))
# trainデータとtestデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    random_state = 0,
    stratify = None,
    test_size = 0.2
)
del X,Y
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# データ型の変換＆正規化
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# trainデータからvalidデータを分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    random_state = 0,
    test_size = 0.2
)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape) 

base_model = Xception(
    include_top = False,
    weights = "imagenet",
    input_shape = None
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(1)(x)

datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False
)
# EarlyStopping
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    verbose = 1
)

# ModelCheckpoint
weights_dir = './weights_age/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.weights.h5",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    save_freq = 10
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 3,
    verbose = 1
)

# log for TensorBoard
logging = TensorBoard(log_dir = "log_age/")

# RMSE
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis = -1)) 

# ネットワーク定義
model = Model(inputs = base_model.input, outputs = predictions)

#108層までfreeze
for layer in model.layers[:108]:
    layer.trainable = False

    # Batch Normalizationのfreeze解除
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True
    if layer.name.endswith('bn'):
        layer.trainable = True

#109層以降、学習させる
for layer in model.layers[108:]:
    layer.trainable = True
    
# layer.trainableの設定後にcompile
model.compile(
    optimizer = Adam(),
    loss = root_mean_squared_error,
    metrics=['accuracy']
)

#%%time
hist = model.fit(
    datagen.flow(X_train, y_train, batch_size = 64),
    steps_per_epoch = X_train.shape[0] //64,
    epochs = 50,
    validation_data = (X_valid, y_valid),
    callbacks = [early_stopping, reduce_lr],
    shuffle = True,
    verbose = 1
)
print(hist.history.keys())
plt.figure(figsize = (18,6))

# accuracy
plt.subplot(1, 2, 1)
plt.plot(hist.history["accuracy"], label = "acc", marker = "o")
plt.plot(hist.history["val_accuracy"], label = "val_acc", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("accuracy")
#plt.title("")
plt.legend(loc = "best")
plt.grid(color = 'gray', alpha = 0.2)

# loss
plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"], label = "loss", marker = "o")
plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("loss")
#plt.title("")
plt.legend(loc = "best")
plt.grid(color = 'gray', alpha = 0.2)

plt.savefig("accuracy_loss_age.png")

score = model.evaluate(X_test, y_test, verbose = 1)
print("evaluate loss: {[0]:.4f}".format(score))
print("evaluate acc: {[1]:.1%}".format(score))

model_dir = './model_age/'
if os.path.exists(model_dir) == False:os.mkdir(model_dir)

model.save(model_dir + 'model.hdf5')

# optimizerのない軽量モデルを保存（学習や評価は不可だが、予測は可能）
model.save(model_dir + 'model-opt.hdf5', include_optimizer = False)

# testデータ30件の予測確率
preds = model.predict(X_test[0:30])

# testデータ30件の画像と予測ラベル・予測確率を出力
plt.figure(figsize = (16, 6))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.axis("off")
    pred = round(preds[i][0],1)
    true = y_test[i]
    if abs(pred - true) < 5.4:
        plt.title(str(true) + '\n' + str(pred))
    else:
        plt.title(str(true) + '\n' + str(pred), color = "red")
    plt.imshow(X_test[i])
plt.savefig("result_age.png")
