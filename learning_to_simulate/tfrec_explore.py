import collections
import functools
import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import reading_utils
import tree

from learning_to_simulate import learned_simulator
from learning_to_simulate import noise_utils
from learning_to_simulate import reading_utils
import md

def make_dict_tensor(t,p,num):
    t_tensor = tf.convert_to_tensor(t)
    p_tensor = tf.convert_to_tensor(p)

    type_dict = {
      "particle_type": t_tensor,
      "key": tf.convert_to_tensor(num),
    }
    pos_dict = {
      "position": p_tensor
    }
    return type_dict, pos_dict

# making TF dataset
# num_trajectory = 3
# all_dat = [];
# for i in range(0,num_trajectory - 1):
#     t,p = md.getDataFrames(i)
#     all_dat.append(make_dict_tensor(t,p,i))
# ds = tf.data.Dataset.from_tensor_slices(all_dat)



example_path = os.path.join('/private/tmp/datasets', "example.tfrecords")
# np.random.seed(0)
# with tf.io.TFRecordWriter(example_path) as file_writer:
#     for i in range(1, 1):
#         x, y = md.getDataFrames(i)
#         record_bytes = tf.train.Example(features=tf.train.Features(feature={
#          "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
#          "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
#         })).SerializeToString()
#         file_writer.write(record_bytes)
#
# print(type(record_bytes))
# ds = tf.data.TFRecordDataset(record_bytes)
# print(type(ds))

ds = tf.data.TFRecordDataset(example_path)
ds = ds.map(functools.partial(
    reading_utils.parse_serialized_simulation_example, metadata=metadata))
for example in ds.take(1):
    print(example)