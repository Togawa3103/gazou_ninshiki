
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

dir_path="test"
#%%time
# ZIP�ǂݍ���
imgfiles = os.listdir(dir_path)
#print(z.namelist())
# �摜�t�@�C���p�X�̂ݎ擾
#imgfiles = [ x for x in z.namelist()]
#print(imgfiles[0])
#exit()
X = []
Y = []
for imgfile in imgfiles:
    #print(imgfile)
    #exit()
    # ZIP����摜�ǂݍ���
    image = Image.open(dir_path+"/"+imgfile)
    # RGB�ϊ�
    image = image.convert('RGB')
    # ���T�C�Y
    image = image.resize((image_size, image_size))
    # �摜����z��ɕϊ�
    data = np.asarray(image)
    file = os.path.basename(imgfile)
    #print(file)
    file_split = [i for i in file.split('_')]
    #print(file_split)
    X.append(data)
    Y.append(file_split[1])
    if(file_split[1]=='3'):
        print(imgfile)
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
# train�f�[�^��test�f�[�^�ɕ���
X_train = X.astype('float32') / 255
X_test = X.astype('float32') / 255

model =load_model("./model/model-opt.hdf5")
# test�f�[�^30���̗\�����x��
pred_classes = np.argmax(model.predict(X_test[0:len(X)]), axis = 1)

# test�f�[�^30���̗\���m��
pred_probs = np.max(model.predict(X_test[0:len(X)]), axis = 1)
pred_probs = ['{:.4f}'.format(i) for i in pred_probs]

# test�f�[�^30���̉摜�Ɨ\�����x���E�\���m�����o��
plt.figure(figsize = (16, 6))
for i in range(len(X)):
    plt.subplot(3, 10, i + 1)
    plt.axis("off")
    plt.title(classes[pred_classes[i]]+'\n'+pred_probs[i], color = "red")
    plt.imshow(X_test[i])
plt.savefig("result2.png")

