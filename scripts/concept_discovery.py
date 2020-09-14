from multiprocessing import dummy as multiprocessing
import numpy as np
import os
import tcav.model as model
import tensorflow as tf

from helpers import load_images_from_files

class ConceptDiscovery(object):

  def __init__(self,
               model,
               target_class,
               bottleneck,
               sess,
               source_dir,
               channel_mean=True,
               max_imgs=40,
               min_imgs=20,
               num_discovery_imgs=40,
               num_workers=20):

    self.model = model
    self.bottleneck = bottleneck
    self.sess = sess
    self.target_class = target_class
    self.source_dir = source_dir
    self.channel_mean = channel_mean
    self.image_shape = model.get_image_shape()[:2]
    self.max_imgs = max_imgs
    self.min_imgs = min_imgs
    if num_discovery_imgs is None:
      num_discovery_imgs = max_imgs
    self.num_discovery_imgs = num_discovery_imgs
    self.num_workers = num_workers

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
      raw_imgs, final_filenames = self.load_concept_imgs(self.target_class, self.num_discovery_imgs)
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
