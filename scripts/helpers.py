from multiprocessing import dummy as multiprocessing
import numpy as np
from PIL import Image
import tcav.model as model
import tensorflow as tf

# Helper functions
def map_labels_to_dirs(labels_to_dirs='./labels/ImageNet_label.txt'):

    labels_to_dirs = open(labels_to_dirs)
    mapping = {}
    for line in labels_to_dirs:
        filename, labels = line.strip().split('    ')
        label = labels.split(', ')[0]
        mapping[label] = filename
    
    mapping['crane bird'] = 'n02012849'
    mapping['african grey'] = 'n01817953'
    mapping['tank suit'] = 'n03710721'

    return mapping

def map_images_to_labels(labels_to_dirs='./labels/ImageNet_label.txt'):

    labels_to_dirs = open(labels_to_dirs)
    mapping = {}
    for line in labels_to_dirs:
        filename, labels = line.strip().split('    ')
        label = labels.split(', ')[0]
        mapping[filename] = label

    mapping['n02012849'] = 'crane bird'
    mapping['n01817953'] = 'african grey'
    mapping['n03710721'] = 'tank suit'

    return mapping

def get_acts_from_images(imgs, model, bottleneck_name):
  return np.asarray(model.run_examples(imgs, bottleneck_name)).squeeze()

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

def cosine_similarity(a, b):
  """Cosine similarity of two vectors."""
  assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
  a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
  if a_norm * b_norm == 0:
    return 0.
  cos_sim = np.sum(a * b) / (a_norm * b_norm)
  return cos_sim