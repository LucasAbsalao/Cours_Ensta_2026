# Produit matrice-vecteur v = A.u
import numpy as np
import time

# Dimension du problème (peut-être changé)
dim = 600
# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
print(f"u = {u}")

bufferFilename = f"outputNoParallelisation.txt"
out = open(bufferFilename, 'w')

tries = 10
timeToDotProduct = 0
# Produit matrice-vecteur
for i in range(tries):
    debut = time.time()
    v = A.dot(u)
    fin = time.time()
    timeToDotProduct += (fin-debut)

print(f"v = {v}")
out.write(f"Temps pris pour la éxécution du produit scalaire : {timeToDotProduct/tries} secondes\n")