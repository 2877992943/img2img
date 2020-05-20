import os
import zipfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
#import PIL
import numpy as np

#import tensorflow.compat.v1 as tf
import tensorflow as tf


from tensor2tensor.data_generators.celeba import Img2imgCeleba

@registry.register_problem
class cele1(Img2imgCeleba):
    def preprocess_example(self, example, unused_mode=None, unused_hparams=None):
        print('process', example)
        image = example["inputs"]
        # Remove boundaries in CelebA images. Remove 40 pixels each side
        # vertically and 20 pixels each side horizontally.
        image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)
        image_8 = image_utils.resize_by_area(image, 8)
        image_32 = image_utils.resize_by_area(image, 32)

        example["inputs"] = image_8
        example["targets"] = image_32
        print (example['inputs'])
        print(example['targets'])
        return example
    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        print ('gene')
        train_gen = self.generator(tmp_dir, 162770)
        train_paths = self.training_filepaths(
            data_dir, self.train_shards, shuffled=False)
        generator_utils.generate_files(train_gen, train_paths)

        dev_gen = self.generator(tmp_dir, 19867, 162770)
        dev_paths = self.dev_filepaths(data_dir, self.dev_shards, shuffled=False)
        generator_utils.generate_files(dev_gen, dev_paths)

        test_gen = self.generator(tmp_dir, 19962, 162770 + 19867)
        test_paths = self.test_filepaths(data_dir, self.test_shards, shuffled=False)
        generator_utils.generate_files(test_gen, test_paths)

        generator_utils.shuffle_dataset(train_paths + dev_paths + test_paths)
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


        ### debug
        part_img='../data_cele/celeba-dataset/some_img'
        all_img='../data_cele/celeba-dataset/img_align_celeba/img_align_celeba'
        unzipped_folder=all_img
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
            if img_name not in img_landmarks:continue
            landmarks = img_landmarks[img_name]
            if img_name not in img_attrs: continue
            attrs = img_attrs[img_name]

            #img = PIL.Image.open(filename)#.convert("L")
            #imgarr = np.array(img)
            #exm=self.preprocess_example({'inputs':imgarr}) # 这个方法是训练时候用的
            #yield {'inputs':imgarr}

            with tf.gfile.Open(filename, "rb") as f:
                encoded_image_data = f.read()
                yield {
                    "image/encoded": [encoded_image_data],
                    "image/format": ["jpeg"],
                    "attributes": attrs,
                    "landmarks": landmarks,
                }