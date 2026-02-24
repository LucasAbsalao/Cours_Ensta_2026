__global__ void sum_vector(int dim, float *a, float *b, float *c) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;

    if (ind < dim){
        c[ind] = a[ind] + b[ind];
    }
}