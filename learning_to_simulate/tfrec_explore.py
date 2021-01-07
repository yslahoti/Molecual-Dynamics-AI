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
    return ((type_dict, pos_dict))

# making TF dataset
num_trajectory = 3
all_dat = [];
for i in range(0,num_trajectory - 1):
    t,p = md.getDataFrames(i)
    all_dat.append(make_dict_tensor(t,p,i))
ds = tf.data.Dataset.from_tensor_slices(all_dat)



