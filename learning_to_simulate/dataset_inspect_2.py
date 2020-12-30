import tensorflow as tf
print(tf.version.VERSION)
import json
import os
import functools
from learning_to_simulate import reading_utils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.executing_eagerly()
tf.compat.v1.enable_eager_execution()

data_path = '/private/tmp/datasets/WaterRamps'
INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def input_fn():
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    metadata = json.loads(fp.read())

  ds = tf.data.TFRecordDataset(data_path)
  ds1 = ds.map(functools.partial(
    reading_utils.parse_serialized_simulation_example, metadata=metadata))

  for parsed_record in ds1.take(10):
    print(parsed_record)


raw_dataset = tf.data.TFRecordDataset('/private/tmp/datasets/WaterDropSample/train.tfrecord')

for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  print(example)

feature_description = {
  'key': tf.io.FixedLenFeature([], tf.int64),
  'particle_type': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

parsed_image_dataset = raw_dataset.map(_parse_image_function)
print(parsed_image_dataset)

for feature in parsed_image_dataset:
  image_raw = feature['particle_type'].numpy()
  print(image_raw)