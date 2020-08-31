import os
import tcav.model as model
import tensorflow as tf

from helpers import load_images_from_files

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