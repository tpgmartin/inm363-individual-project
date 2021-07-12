import keras
from keras import applications
from keras.layers import Input
import os
import tensorflow

def save_model_to_file(filename, architecture, weights='imagenet', input_shape=(299, 299, 3)):

    base_path = os.getcwd()
    model_path = os.path.join(base_path,filename)

    # Create and save network file
    if not os.path.exists(model_path):
        # Initialise Keras object
        if architecture == 'inceptionv3':
            mymodel = applications.InceptionV3(
                weights=weights,
                input_tensor=Input(shape=input_shape)
            )
        elif architecture == 'resnet152v2':
            mymodel = applications.ResNet152V2(
                weights=weights,
                input_tensor=Input(shape=input_shape)
            )
        elif architecture == 'inceptionresnetv2':
            mymodel = applications.InceptionResNetV2(
                weights=weights,
                input_tensor=Input(shape=input_shape)
            )
        else:
            print('Invalid architecture')
            return

        mymodel.save(os.path.join(base_path,filename))

if __name__ == '__main__':

    filenames = ['inception_v3.h5', 'resnet152v2_v3.h5', 'inceptionresnetv2_v3.h5']

    for filename in filenames:
        architecture = filename.split('_')[0]
        save_model_to_file(filename, architecture)
