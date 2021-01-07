import MDAnalysis as mda
from MDAnalysis.lib.formats.libdcd import DCDFile
import tensorflow.compat.v1 as tf

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
      "key": tf.convert_to_tensor(num),
    }
    pos_dict = {
      "position": p_tensor
    }
    return ((type_dict, pos_dict))

# making TF dataset
num_trajectory = 3
all_dat = [];
for i in range(1,num_trajectory+1):
    t,p = getDataFrames(i)
    all_dat.append(make_dict_tensor(t,p,i))
print(all_dat)
ds = tf.data.Dataset.from_tensor_slices(all_dat)





