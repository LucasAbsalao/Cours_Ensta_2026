# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI
import time

def assembleLocalMatrix( A, ibeg : int, iend : int ):
    assert(iend>ibeg)
    u = A[ibeg:iend, :]
    return u


comGlobal = MPI.COMM_WORLD.Dup()
rank      = comGlobal.rank
nbp       = comGlobal.size

bufferFilename = f"outputLine{rank:03d}.txt"
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

ALoc = assembleLocalMatrix(A, ibeg, iend)

glob_array = np.empty(dim, dtype=np.float64)


# Produit matrice-vecteur
debut = time.perf_counter()
v = ALoc.dot(u)
comGlobal.Allgather(([v, MPI.DOUBLE]), [glob_array, MPI.DOUBLE])
fin = time.perf_counter()

local_time = fin-debut
global_time = comGlobal.allreduce(local_time, op=MPI.MAX)

out.write(f"{ALoc} * {u} = {v}\n")
out.write(f"sum = {glob_array}\n")
out.write(f"Temps pris pour la éxécution du produit scalaire local : {fin-debut} secondes\n")
temp_sequentiel = 0.0002040863037109375 #test séquentielle exécuté dans le fichier matvec.py
out.write(f"Speed-up global estimé avec {nbp} processus = {(temp_sequentiel/global_time)}")
out.close()
