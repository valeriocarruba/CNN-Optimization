# -*- coding: utf-8 -*-
"""
# ===========================================================================
# ===========================================================================
# !==   Safwan ALJBAAE, Valerio Carruba                                    ==
# !==   November 2020                                                      ==
# ===========================================================================
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from PIL import Image
from sys import getsizeof
import copy
import time
import datetime
import tracemalloc

class_names = ['circulation state', 'switching orbits', 'libration state']

# read data
def read_data(filename, Images_loc, Labels, set_type):
    data_ast = pd.read_csv(filename,
                           skiprows=0,
                           header=None,
                           delim_whitespace=True,
                           index_col=None,
                           names = Labels,
                           low_memory=False,
                           dtype={'id': np.integer,
                                  'a': np.float64,
                                  'e': np.float64,
                                  'sin_i': np.float64,
                                  'Mag': np.float64,
                                  'label': np.integer
                                  }
                           )
    n_lines =int(len(data_ast))
    data= data_ast.iloc[0:n_lines, :]
    data_id = list(data_ast.id)
    img = [Images_loc + str("{:07d}".format(ast_id)) + '.png' for ast_id in data_id]
    width, height = Image.open(img[0]).convert('1').size
    images = [np.array(Image.open(x).convert('1').getdata()).reshape(width, height) for x in img]
    im_labels = data_ast.label

    print(f'The size of the variable images is : {getsizeof(images)} bytes')
    print(f'We have {len(images)} images ({width} X {height} pixels) in the ', set_type, ' set, belonging to '
          f'{len(set(im_labels))} classes:')
    for i in range(len(set(im_labels))):
        print(f'   {len([x for x in im_labels if x == i])} asteroids in {class_names[i]} (label: {i})')
        print()
    return images, im_labels, data, data_id

filename_train = './TRAINING/m12_training'
Images_loc_train = './TRAINING/fig_res_'
names_train = ['id', 'a', 'e', 'sin_i', 'Mag','label']
set_type = 'training'

train_images, train_labels, train_data, train_id = read_data(filename_train,Images_loc_train, names_train, set_type)

min_pixel = min(list(map(min, train_images[0])))
max_pixel = max(list(map(max, train_images[0])))
print(f'The pixel values of each image vary from {min_pixel} to {max_pixel}')

filename_test = './TEST/m12_test'
Images_loc_test = './TEST/fig_res_'
names_test = ['id', 'a', 'e', 'sin_i', 'Mag', 'label']
set_type = 'testing'

test_images, test_labels, test_data, test_id = read_data(filename_test,Images_loc_test, names_test, set_type)

filename_val = './VALIDATION/m12_validation'
Images_loc_val = './VALIDATION/fig_res_'
names_val = ['id', 'a', 'e', 'sin_i', 'Mag','label']
set_type = 'validation'

val_images, val_labels, val_data, val_id = read_data(filename_val, Images_loc_val, names_val, set_type)

# preprocessing the data: rescale the pixels value to range from 0 to 1
train_images = train_images / max_pixel
test_images = test_images / max_pixel
val_images = val_images / max_pixel

# Define data augmentation
data_augmentation = keras.Sequential(
    [tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(100,100,1)),
     tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
     tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
     ]
)

# Set up the VGG model
def define_model():
    model = Sequential()
#    Data augmentation, uncomment if you wish to use it
#    model.add(data_augmentation)
    model.add(Conv2D(train_images.shape[1], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100,1)))
    model.add(Conv2D(train_images.shape[1], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#    Batch Normalization, uncomment if you wish to use it
#    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
#    Dropout, uncomment if you wish to use it
#    model.add(Dropout(0.2))
    model.add(Flatten(input_shape=(train_images.shape[1], train_images.shape[2]))),
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(64, activation=tf.nn.relu))
    model.add(Dense(3, activation=tf.nn.softmax))
    opt = SGD(learning_rate=0.001, momentum=0.9)
    return model

# Set up the model Inception, uncomment if you wish to use it
#def define_model():
#    model = Sequential()
#    model.add(data_augmentation)
#    model.add(Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100,1)))
#    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100,1)))
#    model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
#    model.add(BatchNormalization())
#    model.add(MaxPooling2D((3, 3)))
#    model.add(Dropout(0.2))
#    model.add(Flatten())
#    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#    model.add(Dense(3, activation='softmax'))
#    # compile model
#    return model

# Set up the model ResNet, uncomment if you wish to use it
#def define_model():
#    model = Sequential()
#    model.add(data_augmentation)
#    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100,1)))
#    model.add(Conv2D(64, (3, 3), activation='linear', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100,1)))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
#    model.add(Flatten())
#    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#    model.add(Dense(3, activation='softmax'))
#    return model

model=define_model()
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

start_time = time.time()
# starting the monitoring
tracemalloc.start()

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

# This checkpoint object will store the model parameters
# in the file "weights.hdf5"
checkpoint = ModelCheckpoint('./weights.hdf5',
                             save_weights_only=True,
                             monitor='accuracy',
                             mode='max',
                             verbose=1,
                             save_best_only=True,)

# fit the model, using the checkpoint as a callback
x = model.fit(train_images, train_labels, epochs=20, callbacks=[checkpoint],
              verbose = 0, validation_data=(val_images, val_labels))
model.load_weights('./weights.hdf5')

end_time = time.time()
exec_time = datetime.timedelta(seconds=(end_time - start_time))
print(f'\n --- The execution time was: {exec_time} (h:m:s) ---')
# displaying the memory: The output is given in form of (current, peak), i.e, current memory is the memory the code is currently using, Peak memory is the maximum space the program used while executing.
print(tracemalloc.get_traced_memory())

model.save("model_VGG")
# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("model_VGG")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_images), reconstructed_model.predict(test_images)
)

# plot diagnostic learning curves
def summarize_diagnostics(x):
    fig = plt.figure()
    figure = fig.add_subplot(211)
    figure.plot(x.epoch, x.history['loss'], color='blue', label='train')
    figure.plot(x.epoch, x.history['val_loss'],color='orange', label='val')
    #plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Cross Entropy Loss: VGG')
    figure = fig.add_subplot(212)
    figure.plot(x.epoch, x.history['accuracy'], color='blue', label='train')
    figure.plot(x.epoch, x.history['val_accuracy'],color='orange', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Classification Accuracy: VGG')
    #plt.show()
    plt.tight_layout()
    fig.savefig('history_model_VGG.png', format='png', dpi=300)
    plt.close(fig)

summarize_diagnostics(x)

predictions = model.predict(test_images)
predict_label = [int(np.argmax(x)) for x in predictions]
predict_acc = [100 * max(x) for x in predictions]

predicted_data = copy.deepcopy(test_data)
predicted_data['predicted_label'] = list(predict_label)
predicted_data.to_csv(r'm12_pred_data.csv', index=False, header=False, sep=' ', float_format='%.7f')
print()


# show the first images in the test data
def show_images():
    fig = plt.figure(figsize=(8, 12))
    for i in range(50):
        plt.subplot(10, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i])
        color = 'blue'
        plt.xlabel("{} ({:2.0f}%)".format(predict_label[i], predict_acc[predict_label[i]]),color=color, fontsize=10)
        plt.ylabel("{}".format(test_id[i]), color=color, fontsize=10)
    plt.subplots_adjust(hspace=0.3, wspace=0)
    # plt.show()
    fig.savefig('m12_predicted_data.png', format='png', dpi=300)
    plt.close(fig)

show_images()

print("End")
