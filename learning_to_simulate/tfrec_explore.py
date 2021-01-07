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

def getData_tensor(t,p):
    t_tensor = tf.convert_to_tensor(t)
    p_tensor = tf.convert_to_tensor(p)
    return t_tensor, p_tensor

def make_dict(t,p,num):
    type_dict = {
      "particle_type": t,
      "key": tf.convert_to_tensor(num),
    }
    pos_dict = {
      "position": p
    }
    return ((type_dict, pos_dict))


# for each trajectory, call getData_tensor to convert to tensor then make_dict
# to get tuple dictionary. After running on all
