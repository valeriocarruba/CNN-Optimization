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
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
import numpy as np
import pandas as pd
from PIL import Image
from sys import getsizeof
import copy
import time
import datetime
import tracemalloc


class_names = ['circulation state', 'switching orbits', 'libration state','aligned state']
# read data

def read_data(filename, Images_loc, Labels, set_type):
    data_ast = pd.read_csv(filename,
                           skiprows=0,
                           header=None,
                           delim_whitespace=True,
                           index_col=None,
                           names = Labels,
                           low_memory=False,
                           dtype={'id': str,
                                  'a': np.float64,
                                  'e': np.float64,
                                  'sin_i': np.float64,
                                  'Mag': np.float64,
                                  'Label': np.integer,
                                  }
                           )
    n_lines =int(len(data_ast))
    data= data_ast.iloc[0:n_lines, :]
    data_id = list(data_ast.id)
    img = [Images_loc +  ast_id + '.png' for ast_id in data_id]
    width, height = Image.open(img[0]).convert('1').size
    images = [np.array(Image.open(x).convert('1').getdata()).reshape(width, height) for x in img]
    im_labels = data_ast.Label

    print(f'The size of the variable images is : {getsizeof(images)} bytes')
    print(f'We have {len(images)} images ({width} X {height} pixels) in the ', set_type, ' set, belonging to '
          f'{len(set(im_labels))} classes:')
    for i in range(len(set(im_labels))):
        print(f'   {len([x for x in im_labels if x == i])} asteroids in {class_names[i]} (label: {i})')
        print()
    return images, im_labels, data, data_id

filename_test = './MULTI_IMAGES/nu6_status'
Images_loc_test = './MULTI_IMAGES/fig_res_'
names_test = ['id', 'a', 'e', 'sin_i', 'Mag','Label']
set_type = 'testing'

test_images, test_labels, test_data, test_id = read_data(filename_test,Images_loc_test, names_test, set_type)

min_pixel = min(list(map(min, test_images[0])))
max_pixel = max(list(map(max, test_images[0])))

# preprocessing the data: rescale the pixels value to range from 0 to 1
test_images = test_images / max_pixel

start_time = time.time()
# starting the monitoring
tracemalloc.start()

#reconstructed_model = keras.models.load_model("model_VGG")
reconstructed_model = keras.models.load_model("model_VGG_alone")
#reconstructed_model = keras.models.load_model("model_Inception")
#reconstructed_model = keras.models.load_model("model_ResNet")

predictions = reconstructed_model.predict(test_images)
predict_labels = [int(np.argmax(x)) for x in predictions]
predict_acc = [100 * max(x) for x in predictions]

end_time = time.time()
exec_time = datetime.timedelta(seconds=(end_time - start_time))
print(f'\n --- The execution time was: {exec_time} (h:m:s) ---')
# displaying the memory: The output is given in form of (current, peak), i.e, current memory is the memory the code is currently using, Peak memory is the maximum space the program used while executing.
print(tracemalloc.get_traced_memory())

predicted_data = copy.deepcopy(test_data)
predicted_data['predicted_labels'] = list(predict_labels)
predicted_data.to_csv(r'nu6_pred_data.csv', index=False, header=False, sep=' ', float_format='%.7f')
print()

print(accuracy_score(test_labels, predict_labels))
print(fbeta_score(test_labels, predict_labels, average = 'micro', beta=0.5))
print(fbeta_score(test_labels, predict_labels, average = 'macro', beta=0.5))
print(fbeta_score(test_labels, predict_labels, average = 'weighted', beta=0.5))
