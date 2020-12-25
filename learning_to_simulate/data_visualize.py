import json
import os

import tensorflow as tf
import numpy as np
import functools
from learning_to_simulate import reading_utils

print(tf.__version__)


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

tf.compat.v1.enable_eager_execution()

datapath = '/private/tmp/datasets/WaterRamps/valid.tfrecord'
datapath2 = '/private/tmp/datasets/WaterRamps/metadata.json'


ds = tf.data.TFRecordDataset(datapath)
metadata = _read_metadata(datapath2)

ds = ds.map(functools.partial(
    reading_utils.parse_serialized_simulation_example, metadata=metadata))

print(ds)