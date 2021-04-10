import keras
from keras.models import load_model
from multiprocessing import dummy as multiprocessing
import numpy as np
from PIL import Image
import tcav.model as model
import tensorflow as tf
from wrapper import KerasModelWrapper

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
    if '.h5' in model_path:
      mymodel = load_model(model_path)

      endpoints_v3 = dict(
        input=mymodel.inputs[0].name,
        input_tensor=mymodel.inputs[0],
        logit=mymodel.outputs[0].name,
        prediction=mymodel.outputs[0].name,
        prediction_tensor=mymodel.outputs[0])

      mymodel = KerasModelWrapper(sess,
        labels_path, [299, 299, 3], endpoints_v3,
        'InceptionV3_public', (-1, 1))
    else:
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

def similarity(cd, num_random_exp=None, num_workers=25):
  """Returns cosine similarity of all discovered concepts.

  Args:
    cd: The ConceptDiscovery module for discovered conceps.
    num_random_exp: If None, calculates average similarity using all the class's
      random concepts. If a number, uses that many random counterparts.
    num_workers: If greater than 0, runs the function in parallel.

  Returns:
    A similarity dict in the form of {(concept1, concept2):[list of cosine
    similarities]}
  """

  def concepts_similarity(cd, concepts, rnd, bn):
    """Calcualtes the cosine similarity of concept cavs.

    This function calculates the pairwise cosine similarity of all concept cavs
    versus an specific random concept

    Args:
      cd: The ConceptDiscovery instance
      concepts: List of concepts to calculate similarity for
      rnd: a random counterpart
      bn: bottleneck layer the concepts belong to

    Returns:
      A dictionary of cosine similarities in the form of
      {(concept1, concept2): [list of cosine similarities], ...}
    """
    similarity_dic = {}
    for c1 in concepts:
      cav1 = cd.load_cav_direction(c1, rnd, bn)
      for c2 in concepts:
        if (c1, c2) in similarity_dic.keys():
          continue
        cav2 = cd.load_cav_direction(c2, rnd, bn)
        similarity_dic[(c1, c2)] = cosine_similarity(cav1, cav2)
        similarity_dic[(c2, c1)] = similarity_dic[(c1, c2)]
    return similarity_dic

  similarity_dic = {bn: {} for bn in cd.bottlenecks}
  if num_random_exp is None:
    num_random_exp = cd.num_random_exp
  randoms = ['random500_{}'.format(i) for i in np.arange(num_random_exp)]
  concepts = {}
  for bn in cd.bottlenecks:
    concepts[bn] = [cd.target_class, cd.random_concept] + cd.dic[bn]['concepts']
  for bn in cd.bottlenecks:
    concept_pairs = [(c1, c2) for c1 in concepts[bn] for c2 in concepts[bn]]
    similarity_dic[bn] = {pair: [] for pair in concept_pairs}
    def t_func(rnd):
      return concepts_similarity(cd, concepts[bn], rnd, bn)
    if num_workers:
      pool = multiprocessing.Pool(num_workers)
      sims = pool.map(lambda rnd: t_func(rnd), randoms)
    else:
      sims = [t_func(rnd) for rnd in randoms]
    while sims:
      sim = sims.pop()
      for pair in concept_pairs:
        similarity_dic[bn][pair].append(sim[pair])
  return similarity_dic

def save_concepts(cd, concepts_dir):
  """Saves discovered concept's images or patches.

  Args:
    cd: The ConceptDiscovery instance the concepts of which we want to save
    concepts_dir: The directory to save the concept images
  """
  for bn in cd.bottlenecks:
    for concept in cd.dic[bn]['concepts']:
      patches_dir = os.path.join(concepts_dir, bn + '_' + concept + '_patches')
      images_dir = os.path.join(concepts_dir, bn + '_' + concept)
      patches = (np.clip(cd.dic[bn][concept]['patches'], 0, 1) * 256).astype(
          np.uint8)
      images = (np.clip(cd.dic[bn][concept]['images'], 0, 1) * 256).astype(
          np.uint8)
      tf.gfile.MakeDirs(patches_dir)
      tf.gfile.MakeDirs(images_dir)
      image_numbers = cd.dic[bn][concept]['image_numbers']
      image_addresses, patch_addresses = [], []
      for i in range(len(images)):
        image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(
            i + 1, image_numbers[i])
        patch_addresses.append(os.path.join(patches_dir, image_name + '.png'))
        image_addresses.append(os.path.join(images_dir, image_name + '.png'))
      save_images(patch_addresses, patches)
      save_images(image_addresses, images)

def save_images(addresses, images):
  """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
  if not isinstance(addresses, list):
    image_addresses = []
    for i, image in enumerate(images):
      image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
      image_addresses.append(os.path.join(addresses, image_name))
    addresses = image_addresses
  assert len(addresses) == len(images), 'Invalid number of addresses'
  for address, image in zip(addresses, images):
    with tf.gfile.Open(address, 'w') as f:
      Image.fromarray(image).save(f, format='PNG')