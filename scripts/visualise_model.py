import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions



inp = Input((299, 299, 3))
inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=inp, input_shape=(299, 299, 3), pooling='avg')
x = inception.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(5, activation='softmax')(x)

complete_model = Model(inp, out)

complete_model.compile(optimizer='adam', loss='categorical_crossentropy')


img_path = image_dir+'/daisy/9158041313_7a6a102f7a_n.jpg'

img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = complete_model.predict(x)
print(preds)

# 0 is the class index for class 'Daisy' in Flowers dataset
flower_output = complete_model.output[:, 0]
last_conv_layer = complete_model.get_layer('mixed10')

grads = K.gradients(flower_output, last_conv_layer.output)[0]                               # Gradient of output with respect to 'mixed10' layer
pooled_grads = K.mean(grads, axis=(0, 1, 2))                                                # Vector of size (2048,), where each entry is mean intensity of
                                                                                            # gradient over a specific feature-map channel
iterate = K.function([complete_model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

#2048 is the number of filters/channels in 'mixed10' layer
for i in range(2048):                                                                       # Multiplies each channel in feature-map array by "how important this
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]                           # channel is" with regard to the class
        
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)                                                            # Following two lines just normalize heatmap between 0 and 1
heatmap /= np.max(heatmap)

plt.imshow(heatmap)

