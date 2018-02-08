import numpy as np
import glob

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.cross_validation import StratifiedKFold

batch_size = 128
nb_classes = 32
epochs = 15
oKdatagen = True
oKdatacross = False

# HOMUS contains images of 40 x 40 pixels
# input image dimensions for train
img_rows, img_cols = 40, 40

# Set image channels order
K.set_image_data_format('channels_last')

def load_data():
    # Load data from data/HOMUS/train_0, data/HOMUS/train_1,...,data/HOMUS_31 folders from HOMUS images
    image_list = []
    class_list = []
    for current_class_number in range(0,nb_classes):    # Number of class
        for filename in glob.glob('./data/HOMUS/train_{}/*.jpg'.format(current_class_number)):
            im = load_img(filename, grayscale=True, target_size=[img_rows, img_cols])  # this is a PIL image
            image_list.append(np.asarray(im).astype('float32')/255)
            class_list.append(current_class_number)

    n = len(image_list)    # Total examples

    X = np.asarray(image_list).reshape(n, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    Y = np_utils.to_categorical(np.asarray(class_list), nb_classes)

    msk = np.random.rand(len(X)) < 0.9 # Train 90% and Test 10%
    X_train, X_test = X[msk], X[~msk]
    Y_train, Y_test = Y[msk], Y[~msk]

    return X_train, Y_train, X_test, Y_test, input_shape

def load_data_cross():
    image_list = []
    class_list = []
    for current_class_number in range(0,nb_classes):    # Number of class
        for filename in glob.glob('./data/HOMUS/train_{}/*.jpg'.format(current_class_number)):
            im = load_img(filename, grayscale=True, target_size=[img_rows, img_cols])  # this is a PIL image
            image_list.append(np.asarray(im).astype('float32')/255)
            class_list.append(current_class_number)

    n = len(image_list)    # Total examples

    if K.image_data_format() == 'channels_first':
        X = np.asarray(image_list).reshape(n, 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = np.asarray(image_list).reshape(n, img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    Y = np.asarray(class_list)
    Y_old = np_utils.to_categorical(np.asarray(class_list), nb_classes)


    return X, Y, input_shape, Y_old


def cnn_model(input_shape):
    #
    # LeNet-5: Artificial Neural Network Structure
    #

    model = Sequential()

    model.add(Conv2D(128, (5, 5), padding='valid', input_shape = input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (5, 5), padding='valid', input_shape = input_shape))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (4, 4), padding='valid'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(16, (2,2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Activation("relu"))
	#No tocar :)
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

##################################################################################
#                               Main program                                     #
##################################################################################
# the data split between train and test sets
X_train, Y_train, X_test, Y_test, input_shape = load_data()
if oKdatacross == False:
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    model = cnn_model(input_shape)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=3)

    if oKdatagen:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=False)
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=epochs)
    else:
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[early_stopping])
        datagen.fit(X_train)
else:
    X, Y, input_shape, Y_old = load_data_cross()
    print(img_rows, 'x', img_cols, 'image size')
    print(input_shape, 'input_shape')
    print(epochs, 'epochs')
    if oKdatagen:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=False)
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=epochs)
        n_folds = 10
        skf = StratifiedKFold(Y, n_folds = n_folds, shuffle=False)
        Y = np_utils.to_categorical(Y, nb_classes)
        for i, (train, test) in enumerate(skf):
            print(i)
            model = None
            model = cnn_model(input_shape)
            print(model.summary())
            model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='loss', patience=3)
    else:
        n_folds = 10
        skf = StratifiedKFold(Y, n_folds = n_folds, shuffle=False)
        Y = np_utils.to_categorical(Y, nb_classes)
        for i, (train, test) in enumerate(skf):
            print(i)
            model = None
            model = cnn_model(input_shape)
            print(model.summary())
            model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='loss', patience=3)

#
# Results
#
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:{:.2f} accuracy: {:.2f}%'.format(loss, acc*100))
#hasta aquí cross validation

# file name to save model
filename='homus_cnn.h5'

# control de pesos
#save network model
#model.save(filename)
#load neetwork model
#model = load_model(filename)

model_json = model.to_json()
with open("model_best.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model_best.h5")
