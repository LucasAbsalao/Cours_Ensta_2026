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

bufferFilename = f"outputMandelbrotMS/outputMandelbrot{rank:03d}.txt"
out = open(bufferFilename, 'w')

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1026, 1026

if height % nbp != 0:
    if rank == 0: print(f"ERREUR: Le hauteur {height} n'est pas divisible par {nbp} processus.")
    comGlobal.Abort()

Nloc = height//nbp

scaleX = 3./width
scaleY = 2.25/height
convergence = np.empty((Nloc, width), dtype=np.double)

global_array = None
if rank == 0:
    global_array = np.empty((height, width), dtype = np.double)

TAG_WORK = 1
TAG_STOP = 2
TAG_RESULT = 3

if rank == 0: #Logique Maitre
    out.write(f"Démarrage Maître-Esclave avec {nbp-1} esclaves.\n")
    
    global_image = np.empty((height, width), dtype=np.double)
    
    # Compteurs
    rows_computed = 0
    current_row_to_dispatch = 0
    active_workers = 0

    deb = time.time()

    # 1. Envoyer une première ligne à chaque esclave
    for worker_rank in range(1, nbp):
        if current_row_to_dispatch < height:
            comGlobal.send(current_row_to_dispatch, dest=worker_rank, tag=TAG_WORK)
            current_row_to_dispatch += 1
            active_workers += 1
        else:
            # Si il y a plus de workers que de lignes
            comGlobal.send(None, dest=worker_rank, tag=TAG_STOP)

    # 2. Boucle de gestion dynamique
    status = MPI.Status() # Pour savoir qui nous répond
    while active_workers > 0:
        # Recevoir le résultat de n'importe quel esclave (ANY_SOURCE) avec la tag TAG_RESULT
        row_idx, row_data = comGlobal.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        sender = status.Get_source()
        
        # Stocker le résultat
        global_image[row_idx, :] = row_data
        rows_computed += 1

        # Envoyer une nouvelle tâche ou arrêter l'esclave
        if current_row_to_dispatch < height:
            comGlobal.send(current_row_to_dispatch, dest=sender, tag=TAG_WORK)
            current_row_to_dispatch += 1
        else:
            comGlobal.send(None, dest=sender, tag=TAG_STOP)
            active_workers -= 1

    fin = time.time()
    total_time = fin - deb

    out.write(f"Temps du calcul de l'ensemble de Mandelbrot : {total_time:.4f}\n")
    
    temp_sequentiel = 1.6140570640563965 #Temps pris pour une éxécution sequentielle
    speedup = temp_sequentiel / total_time
    out.write(f"Speedup Maître-Esclave : {speedup:.4f}\n")

    deb = time.time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(global_image)*255))
    fin = time.time()
    out.write(f"Temps de constitution de l'image : {fin-deb}\n")
    image.save("outputMandelbrotMS/mandelbrot_ms.png")

else: #Logique Esclave
    row_buffer = np.empty(width, dtype=np.double)
    
    count_lignes = 0
    while True:
        # Attendre un ordre du maître
        task = comGlobal.recv(source=0, tag=MPI.ANY_TAG) 
        
        # Si c'est un ordre d'arrêt ou None 
        if task is None:
            out.write(f"Lignes exécutés par ce processus: {count_lignes}\n");
            break
            
        # Si c'est du travail, 'task' contient l'index de la ligne (y)
        y = task
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            row_buffer[x] = mandelbrot_set.convergence(c, smooth=True)

        count_lignes+=1
        comGlobal.send((y, row_buffer), dest=0, tag=TAG_RESULT) #Resultat