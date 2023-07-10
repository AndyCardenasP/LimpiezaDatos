import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 32
LR = 1e-3
MODEL_NAME = 'Classifier-{}-{}.model'.format(LR, '6conv-basic')

def load_images(directory):
    img_path = directory + os.sep
    images = []
    directories = []
    dir_count = []
    prev_root = ''
    count = 0
    print("Leyendo imágenes de ", img_path)

    for root, dirnames, filenames in os.walk(img_path):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                count += 1
                filepath = os.path.join(root, filename)
                image = cv2.imread(filepath)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                images.append(image)
                if prev_root != root:
                    prev_root = root
                    directories.append(root)
                    dir_count.append(count)
                    count = 0
    dir_count.append(count)

    dir_count = dir_count[1:]
    dir_count[0] = dir_count[0] + 1
    print('Directorios leídos:', len(directories))
    print("Imágenes en cada directorio:", dir_count)
    print('Suma total de imágenes en subdirectorios:', sum(dir_count))

    types = []
    index = 0
    for directory in directories:
        name = directory.split(os.sep)
        print(index, name[len(name)-1])
        types.append(name[len(name)-1])
        index += 1

    labels = []
    index = 0
    for count in dir_count:
        for _ in range(count):
            labels.append(types[index])
        index += 1

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels)
    return X, y

X_train, y_train = load_images(os.path.join(os.getcwd(), 'CarneDataset/train'))
X_test, y_test = load_images(os.path.join(os.getcwd(), 'CarneDataset/test'))

X_train = X_train.astype('float32')
X_train = X_train / 255.0

le = LabelEncoder()
y_train = le.fit_transform(y_train)

train_X, valid_X, train_y, valid_y = train_test_split(X_train, to_categorical(y_train), test_size=0.2, random_state=13)

n_classes_train = len(np.unique(y_train))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(n_classes_train, activation='softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adagrad(lr=LR, decay=LR / 100),
              metrics=['accuracy'])

datagen = ImageDataGenerator(rotation_range=10,
                             zoom_range=[0.95, 1.05],
                             height_shift_range=0.1,
                             shear_range=0.1,
                             channel_shift_range=0.1,
                             brightness_range=[0.95, 1.05])

model.fit(datagen.flow(train_X, train_y, batch_size=200),
          epochs=50,
          verbose=1,
          validation_data=(valid_X, valid_y))

predicted_classes_test = model.predict(X_test)
predicted_classes_test = np.argmax(predicted_classes_test, axis=1)

predicted_classes_train = model.predict(X_train)
predicted_classes_train = np.argmax(predicted_classes_train, axis=1)

print(classification_report(y_test, predicted_classes_test, target_names=le.classes_))
confusion_matrix(y_test, predicted_classes_test)
confusion_matrix(y_train, predicted_classes_train)
