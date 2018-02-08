import numpy as np
import glob

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam, nadam
from keras import backend as K
import tensorflow as tf
import keras
from sklearn.cross_validation import StratifiedKFold

batch_size = 64
nb_classes = 164
epochs = 22

file = open("homus_cross_baseline_no_batch.txt","w")
augmentation = True
cross_validation = True
# HOMUS contains images of 40 x 40 pixels
# input image dimensions for train
img_rows, img_cols = 40, 40


def normal_train(model,X_train,Y_train,X_test,Y_test):
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[early_stopping])
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:{:.2f} accuracy: {:.2f}%'.format(loss, acc*100))
    file.write('Test score:{:.2f} accuracy: {:.2f}%'.format(loss, acc*100))

def data_augmentation_train (model,X_train,Y_train,X_test,Y_test):
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2)

    train_datagen.fit(X_train)

    # fits with data augmentation
    # fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
    model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=0)
    #
    # Results
    #
    #
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:{:.2f} accuracy: {:.2f}%'.format(loss, acc*100))
    file.write('Test score:{:.2f} accuracy: {:.2f}%'.format(loss, acc*100))

def load_data():
    #
    # Load data from data/HOMUS/train_0, data/HOMUS/train_1,...,data/HOMUS_31 folders from HOMUS images
    #
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
    with tf.device('/gpu:0'):
        # first convulucional
        model.add(Conv2D(16, (4, 4), padding='valid', input_shape = input_shape))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (4, 4), padding='valid', input_shape = input_shape))
        model.add(Activation("relu"))
        # first maxPooling
        model.add(MaxPooling2D(pool_size=(2, 2)))


        # second convulucional
        model.add(Conv2D(32, (5, 5), padding='valid'))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (5, 5), padding='valid'))
        model.add(Activation("relu"))
        # second maxPooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # first dropout
        model.add(Dropout(0.2))

    with tf.device('/gpu:1'):
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))


    return model


##################################################################################
# Main program

# the data split between train and test sets
X, Y, input_shape, Y_old = load_data()

#msk = np.random.rand(len(X)) < 0.9 # Train 90% and Test 10%
#X_train, X_test = X[msk], X[~msk]
#Y_train, Y_test = Y[msk], Y[~msk]

#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
print(img_rows, 'x', img_cols, 'image size')
print(input_shape, 'input_shape')
print(epochs, 'epochs')





opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

if cross_validation:
    n_folds = 10
    skf = StratifiedKFold(Y, n_folds=n_folds, shuffle=False)
    Y = np_utils.to_categorical(Y, nb_classes)
    for i, (train, test) in enumerate(skf):
        print(i)
        model = None
        #K.clear_session()
        model = cnn_model(input_shape)
        print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer="nadam", metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='loss', patience=3)
        if augmentation:
            data_augmentation_train(model,X[train],Y[train],X[test],Y[test])
        else:
            normal_train(model,X[train],Y[train],X[test],Y[test])


else:
    msk = np.random.rand(len(X)) < 0.9 # Train 90% and Test 10%
    X_train, X_test = X[msk], X[~msk]
    Y_train, Y_test = Y_old[msk], Y_old[~msk]
    model = cnn_model(input_shape)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer="nadam", metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    # train
    if augmentation:
        data_augmentation_train(model,X_train,Y_train,X_test,Y_test)
    else:
        normal_train(model,X_train,Y_train,X_test,Y_test)

# file name to save model
#filename='homus_cnn.h5'

# save network model
#model.save(filename)

# load neetwork model
#model = load_model(filename)
