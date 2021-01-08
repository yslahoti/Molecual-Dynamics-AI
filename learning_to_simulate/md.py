import MDAnalysis as mda
from MDAnalysis.lib.formats.libdcd import DCDFile
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()


def getDataPlot(i):
    u = mda.Universe('../Datasets/solvate.pdb',
                     '../Datasets/wat' + str(i) + '/wat' + str(i) +'_out.dcd')
    p = u.atoms.positions
#Not sure if this is right type
    p = p/30.0
    t = u.atoms.types
    return t, p

def getDataFrames(i):
    u = mda.Universe('../Datasets/solvate.pdb',
                     '../Datasets/wat' + str(i) + '/wat' + str(i) +'_out.dcd')
    t = u.atoms.types
    c = 0
    l = []
    with DCDFile('../Datasets/wat' + str(i) + '/wat' + str(i) +'_out.dcd') as f:
        for frame in f:
            l.append(frame.xyz.tolist())
    return t,l

def make_dict_tensor(t,p,num):
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
num_trajectory = 1
tt = []
for i in range(1,num_trajectory+1):
    t,p = getDataFrames(i)
    x,y = make_dict_tensor(t,p,i)
    dsx = tf.data.Dataset.from_tensor_slices(x)
    dsy = tf.data.Dataset.from_tensor_slices(y)
    dst = tf.tuple(dsx,dsy)
    print(dst)
    print(type(dst))






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



