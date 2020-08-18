import argparse
import numpy as np
from PIL import Image
import sys
from tcav import utils
import tcav.model as model
import tensorflow as tf

def make_model(sess, model_to_run, model_path, labels_path):

  if model_to_run == 'InceptionV3':
    mymodel = model.InceptionV3Wrapper_public(
        sess, model_saved_path=model_path, labels_path=labels_path)
  elif model_to_run == 'GoogleNet':
    mymodel = model.GoolgeNetWrapper_public(
        sess, model_saved_path=model_path, labels_path=labels_path)
  else:
    raise ValueError('Invalid model name')

  return mymodel

def main(args):

  sess = utils.create_session()
  mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)
  print(mymodel.ends)
  
  image_shape = mymodel.get_image_shape()[:2]
  filename = './reference_image.jpg'
  image = np.array(Image.open(filename).resize(image_shape, Image.BILINEAR))
  image = np.float32(image) / 255.0

  # Move to function
  # Image should contain spatial and channel info, should be mult-channel
  if not (len(image.shape) == 3 and image.shape[2] == 3):
    print('BAD PATH')
    return None
  else:
    print('GOOD PATH')
    # return image
  np.reshape(image, (-1, 224, 224, 3))
  bottleneck_name = 'mixed4c'
  # print(mymodel.get_predictions(image))
  print(mymodel.run_examples(image, bottleneck_name))

def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='./imagenet_labels.txt')
  return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))