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
import tree


from learning_to_simulate import learned_simulator
from learning_to_simulate import noise_utils
from learning_to_simulate import reading_utils


print(tf.__version__)
data_path = '/private/tmp/datasets/WaterRamps/'

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())

def prepare_inputs(tensor_dict):
  """Prepares a single stack of inputs by calculating inputs and targets.
  Computes n_particles_per_example, which is a tensor that contains information
  about how to partition the axis - i.e. which nodes belong to which graph.
  Adds a batch axis to `n_particles_per_example` and `step_context` so they can
  later be batched using `batch_concat`. This batch will be the same as if the
  elements had been batched via stacking.
  Note that all other tensors have a variable size particle axis,
  and in this case they will simply be concatenated along that
  axis.
  Args:
    tensor_dict: A dict of tensors containing positions, and step context (
    if available).
  Returns:
    A tuple of input features and target positions.
  """
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])

  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]

  # Remove the target from the input.
  tensor_dict['position'] = pos[:, :-1]

  # Compute the number of particles per example.
  num_particles = tf.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

  if 'step_context' in tensor_dict:
    # Take the input global context. We have a stack of global contexts,
    # and we take the penultimate since the final is the target.
    tensor_dict['step_context'] = tensor_dict['step_context'][-2]
    # Add an extra dimension for stacking via concat.
    tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
  return tensor_dict, target_position

"""Input function for learning simulation."""
# Loads the metadata of the dataset.
metadata = _read_metadata(data_path)
# Create a tf.data.Dataset from the TFRecord.
ds = tf.data.TFRecordDataset([os.path.join(data_path, 'train.tfrecord')])
ds = ds.map(functools.partial(
    reading_utils.parse_serialized_simulation_example, metadata=metadata))
# Splits an entire trajectory into chunks of 7 steps.
# Previous 5 velocities, current velocity and target.
split_with_window = functools.partial(
  reading_utils.split_trajectory,
  window_length=INPUT_SEQUENCE_LENGTH + 1)
ds = ds.flat_map(split_with_window)
# Splits a chunk into input steps and target steps
ds = ds.map(prepare_inputs)
print(ds)