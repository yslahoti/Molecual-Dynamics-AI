import tensorflow as tf
import numpy as np
import functools
from learning_to_simulate import reading_utils

print(tf.__version__)

tf.compat.v1.enable_eager_execution()

datapath = '/private/tmp/datasets/WaterRamps/valid.tfrecord'

raw_dataset = tf.data.TFRecordDataset(datapath)
ds = raw_dataset

ds = ds.map(functools.partial(
    reading_utils.parse_serialized_simulation_example, metadata=metadata))

