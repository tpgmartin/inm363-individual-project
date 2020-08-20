import argparse
from multiprocessing import dummy as multiprocessing
import numpy as np
import os
from PIL import Image
import sys
from tcav import utils
import tcav.model as model
import tensorflow as tf

# Helper functions
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

def load_image_from_file(filename, shape):

  if not tf.gfile.Exists(filename):
    tf.logging.error('Cannot find file: {}'.format(filename))
    return None
  try:
    img = np.array(Image.open(filename).resize(
        shape, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    img = np.float32(img) / 255.0
    if not (len(img.shape) == 3 and img.shape[2] == 3):
      return None
    else:
      return img

  except Exception as e:
    tf.logging.info(e)
    return None
  return img

def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True, run_parallel=True,
                           shape=(299, 299),
                           num_workers=100):

  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)
  if return_filenames:
    final_filenames = []
  if run_parallel:
    pool = multiprocessing.Pool(num_workers)
    imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
                    filenames[:max_imgs])
    if return_filenames:
      final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
                         if imgs[i] is not None]
    imgs = [img for img in imgs if img is not None]
  else:
    for filename in filenames:
      img = load_image_from_file(filename, shape)
      if img is not None:
        imgs.append(img)
        if return_filenames:
          final_filenames.append(filename)
      if len(imgs) >= max_imgs:
        break

  if return_filenames:
    return np.array(imgs), final_filenames
  else:
    return np.array(imgs)

class ConceptDiscovery(object):

  def __init__(self,
               model,
               target_class,
               sess,
               source_dir,
               max_imgs=40,
               min_imgs=20,
               num_discovery_imgs=40,
               num_workers=20):

    self.model = model
    self.sess = sess
    self.target_class = target_class
    self.source_dir = source_dir
    self.image_shape = model.get_image_shape()[:2]
    self.max_imgs = max_imgs
    self.min_imgs = min_imgs
    if num_discovery_imgs is None:
      num_discovery_imgs = max_imgs
    self.num_discovery_imgs = num_discovery_imgs
    self.num_workers = num_workers

  def load_concept_imgs(self, concept, max_imgs=1000):

    concept_dir = os.path.join(self.source_dir, concept)
    img_paths = [
        os.path.join(concept_dir, d)
        for d in tf.gfile.ListDirectory(concept_dir)
    ]
    return load_images_from_files(
        img_paths,
        max_imgs=max_imgs,
        return_filenames=False,
        do_shuffle=False,
        run_parallel=(self.num_workers > 0),
        shape=(self.image_shape),
        num_workers=self.num_workers)

  def predict(self, discovery_images=None):

    if discovery_images is None:
      raw_imgs = self.load_concept_imgs(
          self.target_class, self.num_discovery_imgs)
      self.discovery_images = raw_imgs

    return self.model.get_predictions(self.discovery_images)

def main(args):

  sess = utils.create_session()
  mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)
  
  cd = ConceptDiscovery(
    mymodel,
    args.target_class,
    sess,
    args.source_dir)

  predictions = cd.predict()
  print(len(predictions))

def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='../ACE/ImageNet')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='./imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='dumbbell')
  return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))