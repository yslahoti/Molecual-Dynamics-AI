import MDAnalysis as mda
u = mda.Universe('../Datasets/solvate.pdb', '../Datasets/wat1/wat1_out.dcd')
print(u)