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
from sklearn.model_selection import StratifiedKFold #data cross

batch_size = 64
nb_classes = 32
epochs = 50

# HOMUS contains images of 40 x 40 pixels
# input image dimensions for train
img_rows, img_cols = 40, 40

# Set image channels order
K.set_image_data_format('channels_last')

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

    return X, Y, input_shape


def cnn_model(input_shape):

    # LeNet-5: Artificial Neural Network Structure

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
# Main program
# the data split between train and test sets
X, Y, input_shape = load_data()

print(X.shape[0], 'train samples')
print(X.shape[0], 'test samples')
print(img_rows, 'x', img_cols, 'image size')
print(input_shape, 'input_shape')
print(epochs, 'epochs')

model = cnn_model(input_shape)
print(model.summary())


####################
# DATA AUMENTATION #
####################
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=false) #para que gire


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=epochs)

# Cross Validation
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, Y)
StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

for train_index, test_index in skf.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[early_stopping])
#
# Results
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:{:.2f} accuracy: {:.2f}%'.format(loss, acc*100))
# End Cross Validation

# file name to save model
filename='homus_cnn.h5'

# control de pesos
#save network model
#model.save(filename)
#load neetwork model
#model = load_model(filename)
