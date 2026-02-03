# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI
import time

def assembleLocalMatrixCol( A, b, ibeg : int, iend : int ):
    assert(iend>ibeg)
    u = A[:, ibeg:iend]
    v = b[ibeg:iend]
    return u, v


comGlobal = MPI.COMM_WORLD.Dup()
rank      = comGlobal.rank
nbp       = comGlobal.size

bufferFilename = f"outputMatVecCol/outputCol{rank:03d}.txt"
out = open(bufferFilename, 'w')

# Dimension du problème (peut-être changé)
dim = 600

if dim%nbp != 0:
    print(f"Must have a number of processes which divides the dimension {dim} of the vectors")
    comGlobal.Abort(-1)

NLoc = dim//nbp
ibeg : int = rank * NLoc
iend : int = (rank+1)*NLoc

# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])

ALoc, uLoc = assembleLocalMatrixCol(A, u, ibeg, iend)

glob_array = np.empty(dim, dtype=np.float64)

# Produit matrice-vecteur
debut = time.perf_counter()
v = ALoc.dot(uLoc)
comGlobal.Allreduce([v, MPI.DOUBLE], [glob_array, MPI.DOUBLE], op=MPI.SUM)
fin = time.perf_counter()

local_time = fin-debut
global_time = comGlobal.allreduce(local_time, op=MPI.MAX)

out.write(f"{ALoc} * {uLoc} = {v}\n")
out.write(f"sum = {glob_array}\n")
out.write(f"Temps pris pour la éxécution du produit scalaire local : {fin-debut} secondes\n")
temp_sequentiel = 0.0002040863037109375 #test séquentielle exécuté dans le fichier matvec.py
out.write(f"Speed-up global estimé avec {nbp} processus = {(temp_sequentiel/global_time)}")
out.close()
