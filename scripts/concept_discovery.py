from multiprocessing import dummy as multiprocessing
import numpy as np
import os
from tcav import cav
import tcav.model as model
import tensorflow as tf

from helpers import get_acts_from_images, load_images_from_files

class ConceptDiscovery(object):

  def __init__(self,
               model,
               target_class,
               bottleneck,
               sess,
               source_dir,
               activation_dir,
               cav_dir,
               num_random_exp=2,
               channel_mean=True,
               max_imgs=40,
               min_imgs=20,
               num_discovery_imgs=40,
               num_workers=20):

    self.model = model
    self.bottleneck = bottleneck
    self.sess = sess
    self.target_class = target_class
    self.num_random_exp = num_random_exp
    self.source_dir = source_dir
    self.activation_dir = activation_dir
    self.cav_dir = cav_dir
    self.channel_mean = channel_mean
    self.image_shape = model.get_image_shape()[:2]
    self.max_imgs = max_imgs
    self.min_imgs = min_imgs
    if num_discovery_imgs is None:
      num_discovery_imgs = max_imgs
    self.num_discovery_imgs = num_discovery_imgs
    self.num_workers = num_workers
    self.discovery_images = None

  def load_concept_imgs(self, concept, max_imgs=1000):

    # concept_dir = os.path.join(self.source_dir, concept)
    img_paths = [
        os.path.join(self.source_dir, d)
        for d in tf.gfile.ListDirectory(self.source_dir)
    ]
    return load_images_from_files(
        img_paths,
        max_imgs=max_imgs,
        return_filenames=True,
        do_shuffle=False,
        run_parallel=False,
        shape=(self.image_shape),
        num_workers=self.num_workers)

  def predict(self, discovery_images=None):

    if discovery_images is None:
      raw_imgs, final_filenames = self.load_concept_imgs(None, self.num_discovery_imgs)
      self.discovery_images = raw_imgs

    return self.model.get_predictions(self.discovery_images), final_filenames

  def get_bn_activations(self, bs=100, channel_mean=None):

    imgs, _ = self.load_concept_imgs(None, self.num_discovery_imgs)

    if channel_mean is None:
      channel_mean = self.channel_mean

    if self.num_workers:
      pool = multiprocessing.Pool(self.num_workers)
      output = pool.map(
          lambda i: self.model.run_examples(imgs[i * bs:(i + 1) * bs], self.bottleneck),
          np.arange(int(imgs.shape[0] / bs) + 1))
    else:
      output = []
      for i in range(int(imgs.shape[0] / bs) + 1):
        output.append(
            self.model.run_examples(imgs[i * bs:(i + 1) * bs], self.bottleneck))

    output = np.concatenate(output, 0)
    if channel_mean and len(output.shape) > 3:
      output = np.mean(output, (1, 2))
    else:
      output = np.reshape(output, [output.shape[0], -1])
    return output

  # From ACE code #############################################################
  def _random_concept_activations(self, bottleneck, random_concept):
    """Wrapper for computing or loading activations of random concepts.
    Takes care of making, caching (if desired) and loading activations.
    Args:
      bottleneck: The bottleneck layer name
      random_concept: Name of the random concept e.g. "random500_0"
    Returns:
      A nested dict in the form of {concept:{bottleneck:activation}}
    """
    rnd_acts_path = os.path.join(self.activation_dir, 'acts_{}_{}'.format(
        random_concept, bottleneck))
    if not tf.gfile.Exists(rnd_acts_path):
      # This will load images from source_dir
      rnd_imgs, _ = self.load_concept_imgs(random_concept, self.max_imgs)
      acts = get_acts_from_images(rnd_imgs, self.model, bottleneck)
      with tf.gfile.Open(rnd_acts_path, 'w') as f:
        np.save(f, acts, allow_pickle=False)
      del acts
      del rnd_imgs
    return np.load(rnd_acts_path).squeeze()

  def _calculate_cav(self, c, r, bn, act_c, ow, directory=None):
    """Calculates a sinle cav for a concept and a one random counterpart.
    Args:
      c: concept name
      r: random concept name
      bn: the layer name
      act_c: activation matrix of the concept in the 'bn' layer
      ow: overwrite if CAV already exists
      directory: to save the generated CAV
    Returns:
      The accuracy of the CAV
    """
    if directory is None:
      directory = self.cav_dir
    act_r = self._random_concept_activations(bn, r)
    cav_instance = cav.get_or_train_cav([c, r],
                                        bn, {
                                            c: {
                                                bn: act_c
                                            },
                                            r: {
                                                bn: act_r
                                            }
                                        },
                                        cav_dir=directory,
                                        overwrite=ow)
    return cav_instance.accuracies['overall']

  def _concept_cavs(self, bn, concept, activations, randoms=None, ow=True):
    """Calculates CAVs of a concept versus all the random counterparts.
    Args:
      bn: bottleneck layer name
      concept: the concept name
      activations: activations of the concept in the bottleneck layer
      randoms: None if the class random concepts are going to be used
      ow: If true, overwrites the existing CAVs
    Returns:
      A dict of cav accuracies in the form of {'bottleneck layer':
      {'concept name':[list of accuracies], ...}, ...}
    """
    if randoms is None:
      randoms = [
          'random500_{}'.format(i) for i in np.arange(self.num_random_exp)
      ]
    # if self.num_workers:
    #   pool = multiprocessing.Pool(20)
    #   accs = pool.map(
    #       lambda rnd: self._calculate_cav(concept, rnd, bn, activations, ow),
    #       randoms)
    # else:
    accs = []
    for rnd in randoms:
      accs.append(self._calculate_cav(concept, rnd, bn, activations, ow))
    return accs

  def cavs(self, concept, min_acc=0., ow=True):
    """Calculates cavs for all discovered concepts.
    This method calculates and saves CAVs for all the discovered concepts
    versus all random concepts in all the bottleneck layers
    Args:
      min_acc: Delete discovered concept if the average classification accuracy
        of the CAV is less than min_acc
      ow: If True, overwrites an already calcualted cav.
    Returns:
      A dicationary of classification accuracy of linear boundaries orthogonal
      to cav vectors
    """
    acc = {self.bottleneck: {}}
    concepts_to_delete = []
    # only one bottleneck layer
    # for concept in concepts:
    # for bn in self.bottleneck:
    # for concept in self.dic[bn]['concepts']:
    # concept_imgs = self.dic[bn][concept]['images']
    concept_imgs, _ = self.load_concept_imgs(None, self.num_discovery_imgs)
    concept_acts = get_acts_from_images(concept_imgs, self.model, self.bottleneck)

    acc[self.bottleneck][concept['concept']] = self._concept_cavs(self.bottleneck, concept['concept'], concept_acts, ow=ow)

    if np.mean(acc[self.bottleneck][concept['concept']]) < min_acc:
        concepts_to_delete.append((self.bottleneck, concept))

    if self.discovery_images is None:
      raw_imgs = self.load_concept_imgs(
          self.target_class, self.num_discovery_imgs)
      self.discovery_images = raw_imgs

    print(self.discovery_images)
    print(self.discovery_images)
    target_class_acts = get_acts_from_images(
        self.discovery_images, self.model, self.bottleneck)

    acc[self.bottleneck][self.target_class] = self._concept_cavs(
        self.bottleneck, self.target_class, target_class_acts, ow=ow)

    rnd_acts = self._random_concept_activations(self.bottleneck, self.random_concept)
    acc[self.bottleneck][self.random_concept] = self._concept_cavs(
        self.bottleneck, self.random_concept, rnd_acts, ow=ow)
    
    # TODO: handle concepts to delete

    return acc

    def delete_concept(self, bn, concept):
      """Removes a discovered concepts if it's not already removed.
      Args:
        bn: Bottleneck layer where the concepts is discovered.
        concept: concept name
      """
      self.dic[bn].pop(concept, None)
      if concept in self.dic[bn]['concepts']:
        self.dic[bn]['concepts'].pop(self.dic[bn]['concepts'].index(concept))

    # #########################################################################