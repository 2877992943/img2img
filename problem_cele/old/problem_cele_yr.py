import os
import zipfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry

#import tensorflow.compat.v1 as tf
import tensorflow as tf


from tensor2tensor.data_generators.celeba import Img2imgCeleba

@registry.register_problem
class cele1(Img2imgCeleba):
    def generator(self, tmp_dir, how_many, start_from=0):
        """Image generator for CELEBA dataset.

        Args:
          tmp_dir: path to temporary storage directory.
          how_many: how many images and labels to generate.
          start_from: from which image to start.

        Yields:
          A dictionary representing the images with the following fields:
          * image/encoded: the string encoding the image as JPEG,
          * image/format: the string "jpeg" representing image format,
        """
        # out_paths = []
        # for fname, url in [self.IMG_DATA, self.LANDMARKS_DATA, self.ATTR_DATA]:
        #     path = generator_utils.maybe_download_from_drive(tmp_dir, fname, url)
        #     out_paths.append(path)
        #
        # img_path, landmarks_path, attr_path = out_paths  # pylint: disable=unbalanced-tuple-unpacking
        # unzipped_folder = img_path[:-4]
        # if not tf.gfile.Exists(unzipped_folder):
        #     zipfile.ZipFile(img_path, "r").extractall(tmp_dir)

        unzipped_folder='../data_cele/celeba-dataset/some_img'
        landmarks_path='../data_cele/celeba-dataset/list_landmarks_align_celeba.csv'
        attr_path='../data_cele/celeba-dataset/list_attr_celeba.csv'


        with tf.gfile.Open(landmarks_path) as f:
            landmarks_raw = f.read()

        with tf.gfile.Open(attr_path) as f:
            attr_raw = f.read()

        def process_landmarks(raw_data):
            landmarks = {}
            lines = raw_data.split("\n")
            headings = lines[1].strip().split(',')
            for line in lines[2:-1]:
                values = line.strip().split(',')
                img_name = values[0]
                landmark_values = [int(v) for v in values[1:]]
                landmarks[img_name] = landmark_values
            return landmarks, headings

        def process_attrs(raw_data):
            attrs = {}
            lines = raw_data.split("\n")
            headings = lines[1].strip().split(',')
            for line in lines[2:-1]:
                values = line.strip().split(',')
                img_name = values[0]
                attr_values = [int(v) for v in values[1:]]
                attrs[img_name] = attr_values
            return attrs, headings

        img_landmarks, _ = process_landmarks(landmarks_raw)
        img_attrs, _ = process_attrs(attr_raw)

        image_files = list(sorted(tf.gfile.Glob(unzipped_folder + "/*.jpg")))
        for filename in image_files[start_from:start_from + how_many]:
            img_name = os.path.basename(filename)
            landmarks = img_landmarks[img_name]
            attrs = img_attrs[img_name]

            with tf.gfile.Open(filename, "rb") as f:
                encoded_image_data = f.read()
                yield {
                    "image/encoded": [encoded_image_data],
                    "image/format": ["jpeg"],
                    "attributes": attrs,
                    "landmarks": landmarks,
                }