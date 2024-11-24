
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

#dir_path="part1"
dir_path="test"
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
    #print(file_split)
    X.append(data)
    image.close()
del imgfiles
#print(Y)
#print(Y.count(0),Y.count(1))

X_test = np.array(X)
# データ型の変換＆正規化
X_test = X_test.astype('float32') / 255

# ネットワーク定義
model = load_model("./model_age/model-opt.hdf5")

# testデータ30件の予測確率
preds = model.predict(X_test[0:len(X_test)])

# testデータ30件の画像と予測ラベル・予測確率を出力
plt.figure(figsize = (16, 6))
for i in range(len(X_test)):
    plt.subplot(3, 10, i + 1)
    plt.axis("off")
    pred = round(preds[i][0],1)
    plt.title(str(pred))
    plt.imshow(X_test[i])
plt.savefig("result_age2.png")
