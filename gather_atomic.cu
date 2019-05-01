#include <algorithm>
#include <cstdlib>
#include <sys/time.h>

#include "src/gather_atomic.cuh"
#include "src/parse_data.h"

#define METHOD 2

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

struct timeval t1, t2;
double mytime; 

int main(void) {
    int *input_row_ind_d, *input_col_ind_d, *input_csr_row_d;
    int *input_row_ind_h, *input_col_ind_h, *input_csr_row_h;
    int *total_triangle_d, *total_triangle_h;

    // char file[] = "./data/Theory-3-4-B1k.tsv";
    // int num_nodes = 20;
    // int num_edges = 31;
    
    // char file[] = "./data/Theory-25-81-B1k.tsv";
    // int num_nodes = 2132;
    // int num_edges = 4156;

    // 133321
    char file[] = "./data/Theory-16-25-81-B1k.tsv";
    int num_nodes = 36244;
    int num_edges = 137164;

    // 2102761
    // char file[] = "./data/Theory-25-81-256-B1k.tsv";
    // int num_nodes = 547924;
    // int num_edges = 2132284;
    
    // 66758995
    // char file[] = "./data/Theory-5-9-16-25-81-B1k.tsv";
    // int num_nodes = 2174640;
    // int num_edges = 28667380;

    input_row_ind_h = (int*) malloc(num_edges * sizeof(int));
    input_col_ind_h = (int*) malloc(num_edges * sizeof(int));
    input_csr_row_h = (int*) malloc((num_nodes + 1) * sizeof(int));
    total_triangle_h = (int*) malloc(sizeof(int));

    CUDA_RT_CALL( cudaMalloc((void **)&input_row_ind_d, num_edges * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&input_col_ind_d, num_edges * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&input_csr_row_d, (num_nodes + 1) * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&total_triangle_d, sizeof(int)));

    read_tsv(input_row_ind_h, input_col_ind_h, input_csr_row_h, num_nodes + 1, num_edges, file);

    CUDA_RT_CALL( cudaMemcpy(input_row_ind_d, input_row_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_RT_CALL( cudaMemcpy(input_col_ind_d, input_col_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_RT_CALL( cudaMemcpy(input_csr_row_d, input_csr_row_h, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice) );

    int num_blocks = ceil((num_edges * 1.0) / BLOCK_SIZE);

    gettimeofday(&t1, 0);

#if METHOD == 0
    gather_atomic_naive<<<num_blocks,BLOCK_SIZE>>>(num_edges, input_row_ind_d, input_col_ind_d, input_csr_row_d, total_triangle_d);
#elif METHOD == 1
    gather_atomic_block<<<num_blocks,BLOCK_SIZE>>>(num_edges, input_row_ind_d, input_col_ind_d, input_csr_row_d, total_triangle_d);
#elif METHOD == 2
    gather_atomic_warp<<<num_blocks,BLOCK_SIZE>>>(num_edges, input_row_ind_d, input_col_ind_d, input_csr_row_d, total_triangle_d);
#endif 

    CUDA_RT_CALL( cudaDeviceSynchronize() ); 

    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    
    printf("Time elapsed: %3.5f ms\n", mytime); 

    CUDA_RT_CALL( cudaMemcpy(total_triangle_h, total_triangle_d, sizeof(int), cudaMemcpyDeviceToHost) );
    printf("Total number of triangle: %d\n", total_triangle_h[0]);

    CUDA_RT_CALL( cudaFree(input_row_ind_d) );
    CUDA_RT_CALL( cudaFree(input_col_ind_d) );
    CUDA_RT_CALL( cudaFree(input_csr_row_d) );
    CUDA_RT_CALL( cudaFree(total_triangle_d) );

    free(input_row_ind_h);
    free(input_col_ind_h);
    free(input_csr_row_h);
    free(total_triangle_h);

    return 0;
}
