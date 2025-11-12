import numpy as np, imageio, glob
files = sorted(glob.glob("att_faces/*/*.pgm"))
X = np.column_stack([imageio.imread(f).ravel() for f in files])
np.savetxt("orl_matrix.csv", X, delimiter=",")
