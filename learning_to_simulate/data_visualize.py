import tensorflow as tf
import numpy as np

print(tf.__version__)

tf.compat.v1.enable_eager_execution()

datapath = '/private/tmp/datasets/MultiMaterial/valid.tfrecord'

raw_dataset = tf.data.TFRecordDataset(datapath)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
