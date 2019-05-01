#include <thrust/reduce.h>
#include <sys/time.h>

#include "src/parse_data.h"

#define CUDA_RT_CALL(call)                                                              \
{                                                                                       \
    cudaError_t cudaStatus = call;                                                      \
    if (cudaSuccess != cudaStatus)                                                      \
        fprintf(stderr,                                                                 \
                "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                "with "                                                                 \
                "%s (%d).\n",                                                           \
                #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
}


__global__ static void set_intersect(int *__restrict__ triangleCounts, int* edgeSrc, 
                                int* edgeDst, int* rowPtr, int numEdges) {

    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < numEdges){

        int u = edgeSrc[t];
        int v = edgeDst[t];

        int u_ptr = rowPtr[u];
        int v_ptr = rowPtr[v];

        int u_end = rowPtr[u + 1];
        int v_end = rowPtr[v + 1];

        int w1 = edgeDst[u_ptr];
        int w2 = edgeDst[v_ptr];

        while (u_ptr < u_end && v_ptr < v_end) {
            if (w1 < w2) {
                w1 = edgeDst[++u_ptr];
            }
            else if (w2 < w1) {
                w2 = edgeDst[++v_ptr];
            }
            else {
                w1 = edgeDst[++u_ptr];
                w2 = edgeDst[++v_ptr];
                triangleCounts[t]++;
            }
        }
    }
}

int main() {

    int *input_row_ind_d, *input_col_ind_d, *input_csr_row_d;
    int *input_row_ind_h, *input_col_ind_h, *input_csr_row_h;
    int *output_d, *output_h; 
    struct timeval t1, t2;
    double mytime;

    // char file[] = "./data/Theory-3-4-B1k.tsv";
    // int num_nodes = 20;
    // int num_edges = 31;
    
    // char file[] = "./data/Theory-25-81-B1k.tsv";
    // int num_nodes = 2132;
    // int num_edges = 4156;

    // 133321
    // char file[] = "./data/Theory-16-25-81-B1k.tsv";
    // int num_nodes = 36244;
    // int num_edges = 137164;

    // 2102761
    // char file[] = "./data/Theory-25-81-256-B1k.tsv";
    // int num_nodes = 547924;
    // int num_edges = 2132284;
    
    // 66758995
    char file[] = "./data/Theory-5-9-16-25-81-B1k.tsv";
    int num_nodes = 2174640;
    int num_edges = 28667380;
    
    CUDA_RT_CALL( cudaMalloc((void **)&input_row_ind_d, num_edges * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&input_col_ind_d, num_edges * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&input_csr_row_d, (num_nodes + 1) * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&output_d, num_edges * sizeof(int)) );

    input_row_ind_h = (int*) malloc(num_edges * sizeof(int));
    input_col_ind_h = (int*) malloc(num_edges * sizeof(int));
    input_csr_row_h = (int*) malloc((num_nodes + 1) * sizeof(int));
    output_h = (int*) malloc((num_edges * sizeof(int)));
    
    CUDA_RT_CALL( cudaMemset(output_d, 0, num_edges * sizeof(int)));

    int edges = read_tsv(input_row_ind_h, input_col_ind_h, input_csr_row_h, num_nodes + 1, num_edges, file);
    CUDA_RT_CALL( cudaMemcpy(input_row_ind_d, input_row_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_RT_CALL( cudaMemcpy(input_col_ind_d, input_col_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_RT_CALL( cudaMemcpy(input_csr_row_d, input_csr_row_h, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice) );

    dim3 dimBlock(128);
    dim3 dimGrid(ceil(num_edges / 128.0));
    gettimeofday(&t1, 0);
    set_intersect<<<dimGrid, dimBlock>>>(output_d, input_row_ind_d, input_col_ind_d, input_csr_row_d, num_edges);
    CUDA_RT_CALL( cudaDeviceSynchronize() ); 

    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in real work: %3.5f ms\n", mytime); 
    
    gettimeofday(&t1, 0);
    int total_tc = thrust::reduce(thrust::device, output_d, output_d + num_edges);
    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in reduce: %3.5f ms\n", mytime); 
    printf("Total number of triangle: %d\n", total_tc); 

    // CUDA_RT_CALL( cudaMemcpy(output_h, output_d, num_edges * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL( cudaFree(input_row_ind_d) );
    CUDA_RT_CALL( cudaFree(input_col_ind_d) );
    CUDA_RT_CALL( cudaFree(input_csr_row_d) );
    CUDA_RT_CALL( cudaFree(output_d) );

    free(input_row_ind_h);
    free(input_col_ind_h);
    free(input_csr_row_h);
    free(output_h);
}