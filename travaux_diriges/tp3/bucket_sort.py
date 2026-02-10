import numpy as np
import time
from mpi4py import MPI
import sys
from math import sqrt, log2

out  = None
DEBUG= 0

globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank
name    = MPI.Get_processor_name()

N = 40

if len(sys.argv) > 1:
    N = int(sys.argv[1])

filename = f"Output{rank:03d}.txt"
out      = open(filename, mode='w')

Nloc = N//nbp

if N%nbp != 0:
    print("Le nombre de dimensions doit être divisible par le nombre de processus !")
    globCom.Abort(-1)

out.write(f"Nombre de valeurs locales : {Nloc}\n")

values = np.random.randint(0, 40, size=Nloc, dtype=np.int32)
out.write(f"Valeurs initiales : {values}\n")

debut = time.time()
status = MPI.Status()
values.sort()
out.write(f"Valeurs initiales : {values}\n")
 
local_quantile = np.quantile(values, np.linspace(1/nbp,1,nbp)[:-1], method = 'closest_observation')
out.write(f'local quantiles: {local_quantile}\n')

glob_quantile = np.empty((nbp-1)*nbp, dtype = np.int32)

globCom.Allgather([local_quantile, MPI.INT32_T], [glob_quantile, MPI.INT32_T])
glob_quantile.sort()
out.write(f"global quantiles: {glob_quantile}\n")

scatters = [glob_quantile[i*(nbp-1)] for i in range(1,nbp)]
out.write(f"scatters: {scatters}\n")

counts = np.zeros(nbp, np.int32)
limit = 0
for v in values: #Get how many values of this local arrays are present in each bucket
    while limit < nbp - 1 and v > scatters[limit]:
        limit += 1
    counts[limit] += 1

# Another way to do this
# indices = np.searchsorted(scatters, values, side='right') #Another way to do it
# counts = np.bincount(indices, minlength=nbp)
out.write(f"Counts: {counts}\n")

#Getting the quantity of values that will be received by each process
counts_in_this_bucket = np.zeros(nbp, np.int32)
globCom.Alltoall(counts, counts_in_this_bucket)
out.write(f"Counts in this bucket: {counts_in_this_bucket}\n")

#setting the counts and displacement indexes to be sent
sendCounts = counts.astype(np.int32)
sendDispls = np.insert(np.cumsum(sendCounts)[:-1], 0, 0)

#Setting the displacements and the vector that will be received by each processus
rcvDispls = np.insert(np.cumsum(counts_in_this_bucket)[:-1], 0, 0)
total_rcv = np.sum(counts_in_this_bucket)
rcv_values = np.empty(total_rcv, dtype=np.int32)

#Sending and receiving the data
globCom.Alltoallv([values,(sendCounts, sendDispls), MPI.INT32_T],
                  [rcv_values, (counts_in_this_bucket, rcvDispls), MPI.INT32_T])


out.write(f"Received values for this bucket: {rcv_values}\n")


rcv_values.sort()
out.write(f"Bucket values sorted: {rcv_values}\n")

all_rcv_counts = None
if rank == 0:
    all_rcv_counts = np.empty(nbp, dtype=np.int32)
#Get how many values the rank 0 process will gather of each process
globCom.Gather(np.array([total_rcv], dtype=np.int32), all_rcv_counts, root=0)

# Create the vector that will receive all the values and the displaceme=ents
final_values = None
final_displs = None
if rank == 0:
    final_values = np.empty(N, dtype=np.int32)
    final_displs = np.insert(np.cumsum(all_rcv_counts)[:-1], 0, 0)

# Gatherv used to create the final vector already sorted, gathering the srted vector of each bucket
globCom.Gatherv(rcv_values, [final_values, (all_rcv_counts, final_displs), MPI.INT32_T], root=0)

if rank == 0:
    out.write(f"\n\nOrdered vector: {final_values}\n")