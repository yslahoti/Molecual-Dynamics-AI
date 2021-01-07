import MDAnalysis as mda
from MDAnalysis.lib.formats.libdcd import DCDFile


def getDataPlot(i):
    u = mda.Universe('../Datasets/solvate.pdb',
                     '../Datasets/wat' + str(i) + '/wat' + str(i) +'_out.dcd')
    p = u.atoms.positions
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
            l.append(frame.xyz)
    return l
getDataFrames(1)






