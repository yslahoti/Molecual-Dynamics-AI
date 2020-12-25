import os
import tensorflow as tf
from tfrecord_lite import decode_example

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.enable_eager_execution()

datapath = '/private/tmp/datasets/WaterRamps/valid.tfrecord'

raw_dataset = tf.data.TFRecordDataset(datapath)
raw_dataset