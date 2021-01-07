import MDAnalysis as mda

def getData():
    u = mda.Universe('../Datasets/solvate.pdb', '../Datasets/wat1/wat1_out.dcd')
    u.transfer_to_memory(verbose=True)
    p = u.atoms.positions
    t = u.atoms.types
    return t, p
