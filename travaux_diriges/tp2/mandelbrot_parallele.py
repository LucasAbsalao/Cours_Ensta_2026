# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from mpi4py import MPI
from dataclasses import dataclass
from PIL import Image
from math import log
import time
import matplotlib.cm


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


comGlobal = MPI.COMM_WORLD.Dup()
rank      = comGlobal.rank
nbp       = comGlobal.size

bufferFilename = f"outputMandelbrot/outputMandelbrot{rank:03d}.txt"
out = open(bufferFilename, 'w')

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1026, 1026

if height % nbp != 0:
    if rank == 0: print(f"ERRO: Le hauteur {height} n'est pas divisible par {nbp} processus.")
    comGlobal.Abort()

Nloc = height//nbp
ibeg : int = rank * Nloc
iend : int = (rank+1) * Nloc 

scaleX = 3./width
scaleY = 2.25/height
convergence = np.empty((Nloc, width), dtype=np.double)

global_array = None
if rank == 0:
    global_array = np.empty((height, width), dtype = np.double)

# Calcul de l'ensemble de mandelbrot :
deb = time.time()
for y in range(ibeg, iend):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        convergence[y-ibeg, x] = mandelbrot_set.convergence(c, smooth=True)
fin = time.time()
calc_time = fin - deb

deb = time.time()
comGlobal.Gather(convergence, global_array, root = 0)
fin = time.time()
gather_time = fin - deb

out.write(f"Temps du calcul de l'ensemble de Mandelbrot : {calc_time}\n")
out.write(f"Temps de communication du gather: {gather_time}\n")

# Constitution de l'image résultante :
if rank == 0:
    deb = time.time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(global_array)*255))
    fin = time.time()
    out.write(f"Temps de constitution de l'image : {fin-deb}\n")
    image.save("outputMandelbrot/mandelbrot.png")
    image.show()

calc_global_time = comGlobal.reduce(calc_time, op=MPI.MAX, root=0)
gather_global_time = comGlobal.reduce(gather_time, op=MPI.MAX, root=0)

if rank == 0:
    temp_sequentiel = 1.6140570640563965 #Temps pris pour une éxécution sequentielle
    out.write(f"speedup sans considéré la communication inter processus: {temp_sequentiel/calc_global_time}\n")
    out.write(f"speedup : {temp_sequentiel/(calc_global_time+gather_global_time)}\n")