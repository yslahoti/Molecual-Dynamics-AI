import tensorflow as tf
import numpy as np

print(tf.__version__)
datapath = '/private/tmp/datasets/WaterRamps/valid.tfrecord'

for example in tf.python_io.tf_record_iterator(datapath):
    print(tf.train.Example.FromString(example))
