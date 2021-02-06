import keras
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
import os
import tensorflow as tf

base_path = os.getcwd()
model_path = os.path.join(base_path,'inception_v3.h5')

# Create and save network file
if not os.path.exists(model_path):
    # Initialise Keras object
    mymodel = InceptionV3(
        weights='imagenet',
        input_tensor=Input(shape=(299, 299, 3))
    )

    mymodel.save(os.path.join(base_path,'inception_v3.h5'))
    
