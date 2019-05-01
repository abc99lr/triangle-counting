#include <algorithm>
#include <cstdlib>
#include <sys/time.h>
#include <src/hashmap/concurrent_unordered_map.cuh>
#include <src/groupby/hash/aggregation_operations.cuh>

#include "src/scatter_hash.cuh"
#include "src/parse_data.h"

#define METHOD 0

// #define CUDA_RT_CALL(call)                                                              \
// {                                                                                       \
// 	cudaError_t cudaStatus = call;                                                      \
// 	if (cudaSuccess != cudaStatus)                                                      \
// 		fprintf(stderr,                                                                 \
// 				"ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
// 				"with "                                                                 \
// 				"%s (%d).\n",                                                           \
// 				#call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
// }

struct timeval t1, t2;
double mytime; 

int main(void) {
    int *input_row_ind_d, *input_col_ind_d, *input_csr_row_d;
    int *input_row_ind_h, *input_col_ind_h, *input_csr_row_h;
    int *total_triangle_d, *total_triangle_h;
    // char *byte_map;
    // 12
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

    // // 2102761
    char file[] = "./data/Theory-25-81-256-B1k.tsv";
    int num_nodes = 547924;
    int num_edges = 2132284;
    
    // 66758995
    // char file[] = "./data/Theory-5-9-16-25-81-B1k.tsv";
    // int num_nodes = 2174640;
    // int num_edges = 28667380;

    // printf("unsigned int: %d\n", sizeof(unsigned int)); 
    // printf("unsigned long: %d\n", sizeof(unsigned long)); 
    // printf("unsigned long long: %d\n", sizeof(unsigned long long)); 

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

    // int byte_map_size = num_nodes * num_nodes; 

    // CUDA_RT_CALL( cudaMalloc((void**)&byte_map, byte_map_size * sizeof(char)) ); 
    // CUDA_RT_CALL( cudaMemset(byte_map, 0, byte_map_size) ); 


    int occupancy = 4; 
    map_type *hash_map = new map_type(occupancy * num_edges, op_type::IDENTITY); 

    int num_blocks = ceil((num_edges * 1.0) / BLOCK_SIZE);

    gettimeofday(&t1, 0);
    // Build bit map. 
    // build_bit_map<<<num_blocks, BLOCK_SIZE>>>(num_edges, num_nodes, input_row_ind_d, input_col_ind_d, byte_map);
    build_hash_map<<<num_blocks, BLOCK_SIZE>>>(hash_map, input_row_ind_d, input_col_ind_d, num_edges, num_nodes);
    CUDA_RT_CALL( cudaDeviceSynchronize() ); 
    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in build: %3.5f ms\n", mytime); 

    gettimeofday(&t1, 0);
    // Call scatter kernel. 
#if METHOD == 0 
    scatter_with_hash_map<<<num_blocks, BLOCK_SIZE>>>(num_edges, num_nodes, input_row_ind_d, input_col_ind_d, input_csr_row_d, hash_map, total_triangle_d);
#elif METHOD == 1
    scatter_with_hash_map_block<<<num_blocks, BLOCK_SIZE>>>(num_edges, num_nodes, input_row_ind_d, input_col_ind_d, input_csr_row_d, hash_map, total_triangle_d);
#endif 

    CUDA_RT_CALL( cudaDeviceSynchronize() ); 
    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in real work: %3.5f ms\n", mytime); 

    CUDA_RT_CALL( cudaMemcpy(total_triangle_h, total_triangle_d, sizeof(int), cudaMemcpyDeviceToHost) );
    printf("Total number of triangle: %d\n", total_triangle_h[0]);

    CUDA_RT_CALL( cudaFree(input_row_ind_d) );
    CUDA_RT_CALL( cudaFree(input_col_ind_d) );
    CUDA_RT_CALL( cudaFree(input_csr_row_d) );
    CUDA_RT_CALL( cudaFree(total_triangle_d) );
    // CUDA_RT_CALL( cudaFree(byte_map) );

    free(input_row_ind_h);
    free(input_col_ind_h);
    free(input_csr_row_h);
    free(total_triangle_h);

    return 0;
}
