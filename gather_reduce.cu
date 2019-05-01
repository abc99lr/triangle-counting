#include <thrust/reduce.h>
#include <algorithm>
#include <cstdlib>
#include <sys/time.h>

#include "src/parse_data.h"
#include "src/gather_reduce.cuh"

#define METHOD 0

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

#if METHOD == 1	
    int *input_cm_row_ind_d, *input_cm_col_ind_d, *input_cm_col_d;
    int *input_cm_row_ind_h, *input_cm_col_ind_h, *input_cm_col_h;
#endif

    int *output_value_d;

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

    #if METHOD == 1	
    input_cm_row_ind_h = (int*) malloc(num_edges * sizeof(int));
    input_cm_col_ind_h = (int*) malloc(num_edges * sizeof(int));
    input_cm_col_h = (int*) malloc((num_nodes + 1) * sizeof(int));
#endif 

    CUDA_RT_CALL( cudaMalloc((void **)&input_row_ind_d, num_edges * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&input_col_ind_d, num_edges * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&input_csr_row_d, (num_nodes + 1) * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&output_value_d, num_edges * sizeof(int)));
#if METHOD == 1	
    CUDA_RT_CALL( cudaMalloc((void **)&input_cm_row_ind_d, num_edges * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&input_cm_col_ind_d, num_edges * sizeof(int)) );
    CUDA_RT_CALL( cudaMalloc((void **)&input_cm_col_d, (num_nodes + 1) * sizeof(int)) );
#endif 

    read_tsv(input_row_ind_h, input_col_ind_h, input_csr_row_h, num_nodes + 1, num_edges, file);

    CUDA_RT_CALL( cudaMemcpy(input_row_ind_d, input_row_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_RT_CALL( cudaMemcpy(input_col_ind_d, input_col_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_RT_CALL( cudaMemcpy(input_csr_row_d, input_csr_row_h, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice) );

#if METHOD == 1		
    CUDA_RT_CALL( cudaMemcpy(input_cm_row_ind_d, input_row_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_RT_CALL( cudaMemcpy(input_cm_col_ind_d, input_col_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice));

    gettimeofday(&t1, 0);
    thrust::stable_sort_by_key(thrust::device, input_cm_col_ind_d, input_cm_col_ind_d + num_edges , input_cm_row_ind_d);
    
    CUDA_RT_CALL( cudaMemcpy(input_cm_col_ind_h, input_cm_col_ind_d, num_edges * sizeof(int), cudaMemcpyDeviceToHost));
    make_col_index(input_cm_col_ind_h, input_cm_col_h, num_nodes+1, num_edges);
    CUDA_RT_CALL( cudaMemcpy(input_cm_col_d, input_cm_col_h, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in preparation: %3.5f ms\n", mytime); 
#endif 

    int num_blocks = ceil((num_edges * 1.0) / BLOCK_SIZE);

    gettimeofday(&t1, 0);
#if METHOD == 0
    gather_naive<<<num_blocks,BLOCK_SIZE>>>(num_edges, input_row_ind_d, input_col_ind_d, input_csr_row_d, output_value_d); 
#elif METHOD == 1
    gather_binned<<<num_blocks,BLOCK_SIZE>>>(num_edges, input_row_ind_d, input_col_ind_d, input_csr_row_d,
                                            input_cm_row_ind_d, input_cm_col_ind_d, input_cm_col_d, output_value_d);
#endif
    CUDA_RT_CALL( cudaDeviceSynchronize() ); 													
    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in real work: %3.5f ms\n", mytime); 

    gettimeofday(&t1, 0);
    int total_tc = thrust::reduce(thrust::device, output_value_d, output_value_d + num_edges);
    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in reduce: %3.5f ms\n", mytime); 
    printf("Total number of triangle: %d\n", total_tc); 

    CUDA_RT_CALL( cudaFree(input_row_ind_d) ); 
    CUDA_RT_CALL( cudaFree(input_col_ind_d) ); 
    CUDA_RT_CALL( cudaFree(input_csr_row_d) );
    CUDA_RT_CALL( cudaFree(output_value_d) ); 

    free(input_row_ind_h);
    free(input_col_ind_h);
    free(input_csr_row_h);

#if METHOD == 1
    CUDA_RT_CALL( cudaFree(input_cm_col_ind_d) );
    CUDA_RT_CALL( cudaFree(input_cm_row_ind_d) );
    CUDA_RT_CALL( cudaFree(input_cm_col_d) );

    free(input_cm_col_ind_h);
    free(input_cm_row_ind_h);
    free(input_cm_col_h);
#endif 
    return 0;
}
