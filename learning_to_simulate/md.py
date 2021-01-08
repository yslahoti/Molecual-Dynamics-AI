import MDAnalysis as mda
from MDAnalysis.lib.formats.libdcd import DCDFile
import tensorflow.compat.v1 as tf
import numpy as np
import reading_utils
import train

tf.enable_eager_execution()


def getDataPlot(i):
    u = mda.Universe('../Datasets/solvate.pdb',
                     '../Datasets/wat' + str(i) + '/wat' + str(i) +'_out.dcd')
    p = u.atoms.positions
#Not sure if this is right normalization factor
    p = p/30.0
    t = u.atoms.types
    return t, p

def getDataFrames(i):
    u = mda.Universe('../Datasets/solvate.pdb',
                     '../Datasets/wat' + str(i) + '/wat' + str(i) +'_out.dcd')
    t = u.atoms.types
    t[t == "O"] = 0
    t[t == "H"] = 1
    t = np.asarray(t).astype('int32')

    l = []
    with DCDFile('../Datasets/wat' + str(i) + '/wat' + str(i) +'_out.dcd') as f:
        for frame in f:
            l.append(frame.xyz.tolist())
    return t,l

def make_dict_tensor(t,p):
    t_tensor = tf.convert_to_tensor(t)
    p_tensor = tf.convert_to_tensor(p)

    type_dict = {
      "particle_type": t_tensor,
#      "key": tf.convert_to_tensor(num),
    }
    pos_dict = {
      "position": p_tensor
    }
    return type_dict, pos_dict

# making TF dataset

def getDs(num, str):

    model_input_features = {}
    # Prepare the context features per step.
    model_input_features['particle_type'] = tf.tile(
        tf.expand_dims(context['particle_type'], axis=0),
        [input_trajectory_length, 1])

    if 'step_context' in features:
        global_stack = []
        for idx in range(input_trajectory_length):
            global_stack.append(features['step_context'][idx:idx + window_length])
        model_input_features['step_context'] = tf.stack(global_stack)

    pos_stack = []
    for idx in range(input_trajectory_length):
        pos_stack.append(features['position'][idx:idx + window_length])
    # Get the corresponding positions
    model_input_features['position'] = tf.stack(pos_stack)

    return tf.data.Dataset.from_tensor_slices(model_input_features)

    if (str == "one_step"):
        t,p = getDataFrames(1)
        x,y = make_dict_tensor(t,p)
        s = reading_utils.split_trajectory(x,y)
        print(1)
        for i in range(2,num+1):
            print(i)
            t,p = getDataFrames(i)
            x,y = make_dict_tensor(t,p)
            temp = reading_utils.split_trajectory(x,y)
            s = s.concatenate(temp)
        s = s.map(train.prepare_inputs)
        return s
    elif (str == "rollout"):
        t,p = getDataFrames(1)
        x,y = make_dict_tensor(t,p)
        print(1)
        for i in range(2,num+1):
            print(i)
            t,p = getDataFrames(i)
            x,y = make_dict_tensor(t,p)
        return s

# def getDs(num, str):
#     if (str == "one_step"):
#         t,p = getDataFrames(1)
#         x,y = make_dict_tensor(t,p)
#         s = reading_utils.split_trajectory(x,y)
#         print(1)
#         for i in range(2,num+1):
#             print(i)
#             t,p = getDataFrames(i)
#             x,y = make_dict_tensor(t,p)
#             temp = reading_utils.split_trajectory(x,y)
#             s = s.concatenate(temp)
#         s = s.map(train.prepare_inputs)
#         return s
#     elif (str == "rollout"):
#         t,p = getDataFrames(1)
#         x,y = make_dict_tensor(t,p)
#         print(1)
#         for i in range(2,num+1):
#             print(i)
#             t,p = getDataFrames(i)
#             x,y = make_dict_tensor(t,p)
#         return s


def getDictConcat(num):
    t,p = getDataFrames(1)
    x,y = make_dict_tensor(t,p)
    print(1)
    for i in range(2,num+1):
        print(i)
        t,p = getDataFrames(i)
        t1,t2 = make_dict_tensor(t,p)
        x = {**t1, **x}
        y = {**t2, **y}
    return x,y

x,y = getDictConcat(3)
print(type(x))
print(type(y))
c = 0
for k,v in x.items():
    print(k)
    print(v)
    print(c)
    c+=1











# for example in all_dat:
#     print(type(example))
#     print(len(example))
#     for key, value in example[1].items():
#         print(key)
#         print(value)
#
# #ds = tf.data.Dataset.from_tensor_slices(all_dat)
#
# tf.data.Dataset.from.



