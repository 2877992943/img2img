import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.layers import common_image_attention
from tensor2tensor.models import image_transformer

#import tensorflow.compat.v1 as tf
import tensorflow as tf


def get_feature():
    x = tf.constant(np.random.randn(3, 1, 1, 64),dtype=tf.float32) #[batch,,,] img_cls
    y=tf.constant(np.random.randn(3,7,21,64),dtype=tf.float32)
    rawx=tf.constant(np.random.randint(
        256, size=(3, 1, 1, 1)),dtype=tf.int32)
    rawy=tf.constant(np.random.randint(
        256, size=(3, 7, 7, 3)),dtype=tf.int32)
    inp=tf.constant([0]*1)
    targ=tf.constant([0]*1)
    dic={'inputs':x,
         'targets':y,
         'input_space_id':inp,
         'inputs_raw':rawx,
         'target_space_id':targ,
         'targets_raw':rawy}
    return dic



dic=get_feature()


p1=("ImageTransformerCat",
       image_transformer.Imagetransformer,
       image_transformer.imagetransformer_tiny())
p2=("ImageTransformerDmol",
       image_transformer.Imagetransformer,
       image_transformer.imagetransformerpp_tiny())



if 2>1:
    net=p1[1]
    hparams=p1[2]

    ##
    batch_size = 3
    size = 7
    vocab_size = 256
    p_hparams = problem_hparams.test_problem_hparams(vocab_size,
                                                     vocab_size,
                                                     hparams)
    inputs = np.random.randint(
        vocab_size, size=(batch_size, 1, 1, 1))
    targets = np.random.randint(
        vocab_size, size=(batch_size, size, size, 3))
    with tf.Session() as session:
      features = {
          "inputs": tf.constant(inputs, dtype=tf.int32),

          "targets": tf.constant(targets, dtype=tf.int32),
          "target_space_id": tf.constant(1, dtype=tf.int32),
      }
      model = net(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      #logits, _ = model(features)
      rst=model.body(dic)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)