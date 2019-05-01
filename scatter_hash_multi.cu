#include <thrust/reduce.h>
#include <sys/time.h>
#include <src/hashmap/concurrent_unordered_map.cuh>
#include <src/groupby/hash/aggregation_operations.cuh>

#include "src/parse_data.h"
#include "src/scatter_hash_multi.cuh"

#define NUM_GPUS 2
// #define CUDA_RT_CALL(call)                                                              \
// {                                                                                       \
//     cudaError_t cudaStatus = call;                                                      \
//     if (cudaSuccess != cudaStatus)                                                      \
//         fprintf(stderr,                                                                 \
//                 "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
//                 "with "                                                                 \
//                 "%s (%d).\n",                                                           \
//                 #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
// }

int main() {
    int num_gpus_avail; 
    CUDA_RT_CALL( cudaGetDeviceCount(&num_gpus_avail) ); 
    printf("The system has %d GPU nodes.\n", num_gpus_avail); 
    
    int gpu_ids[NUM_GPUS] = {0, 1}; 
    printf("Going to use %d of them with ids:", NUM_GPUS); 
    for (int i = 0; i < NUM_GPUS; i++) {
        printf(" %d", gpu_ids[i]);
    }
    printf(".\n"); 

    int *input_row_ind_d[NUM_GPUS];
    int *input_col_ind_d[NUM_GPUS];
    int *input_csr_row_d[NUM_GPUS];
    cudaStream_t streams[NUM_GPUS];    
    int *total_triangle_d[NUM_GPUS];
    int *total_triangle_h;

    int *input_row_ind_h, *input_col_ind_h, *input_csr_row_h;
    // int *output_h; 

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
    char file[] = "./data/Theory-25-81-256-B1k.tsv";
    int num_nodes = 547924;
    int num_edges = 2132284;
    
    // 66758995
    // char file[] = "./data/Theory-5-9-16-25-81-B1k.tsv";
    // int num_nodes = 2174640;
    // int num_edges = 28667380;
    
    input_row_ind_h = (int*) malloc(num_edges * sizeof(int));
    input_col_ind_h = (int*) malloc(num_edges * sizeof(int));
    input_csr_row_h = (int*) malloc((num_nodes + 1) * sizeof(int));
    total_triangle_h = (int*) malloc(NUM_GPUS * sizeof(int));

    read_tsv(input_row_ind_h, input_col_ind_h, input_csr_row_h, num_nodes + 1, num_edges, file);

    int num_edges_each[NUM_GPUS];
    int total_edges = 0; 
    int length_common = num_edges / NUM_GPUS; 
    for (int i = 0; i < NUM_GPUS; i++) {
        if (i == NUM_GPUS - 1) {
            num_edges_each[i] = num_edges - total_edges; 
        }
        else {
            num_edges_each[i] = length_common; 
        }
        total_edges += num_edges_each[i]; 
    }

    int occupancy = 4;
    map_type *hash_map[NUM_GPUS];

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 

        CUDA_RT_CALL( cudaMalloc((void **)&input_row_ind_d[i], num_edges * sizeof(int)) );
        CUDA_RT_CALL( cudaMalloc((void **)&input_col_ind_d[i], num_edges * sizeof(int)) );
        CUDA_RT_CALL( cudaMalloc((void **)&input_csr_row_d[i], (num_nodes + 1) * sizeof(int)) );
        CUDA_RT_CALL( cudaMalloc((void **)&total_triangle_d[i], sizeof(int)));
        hash_map[i] = new map_type(occupancy * num_edges, op_type::IDENTITY); 
        CUDA_RT_CALL( cudaStreamCreate(&streams[i]) );
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaMemcpyAsync(input_row_ind_d[i], input_row_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        CUDA_RT_CALL( cudaMemcpyAsync(input_col_ind_d[i], input_col_ind_h, num_edges * sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
        CUDA_RT_CALL( cudaMemcpyAsync(input_csr_row_d[i], input_csr_row_h, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaDeviceSynchronize() ); 
    } 

    gettimeofday(&t1, 0);
    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        dim3 dimBlock(128);
        dim3 dimGrid(ceil(num_edges / 128.0));
        build_hash_map<<<dimGrid, dimBlock, 0, streams[i]>>>(hash_map[i], input_row_ind_d[i], input_col_ind_d[i], num_edges, num_nodes, 0);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaDeviceSynchronize() ); 
    } 

    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in build: %3.5f ms\n", mytime); 
  
    gettimeofday(&t1, 0);
    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        dim3 dimBlock(128);
        dim3 dimGrid(ceil(num_edges_each[i] / 128.0));
        scatter_with_hash_map<<<dimGrid, dimBlock, 0, streams[i]>>>(num_edges_each[i], num_nodes, input_row_ind_d[i] + i * length_common, input_col_ind_d[i], input_csr_row_d[i], hash_map[i], total_triangle_d[i], i * length_common);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaDeviceSynchronize() ); 
    } 

    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in real work: %3.5f ms\n", mytime); 

    int total_tc_final = 0;

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaMemcpyAsync(&total_triangle_h[i], total_triangle_d[i], sizeof(int), cudaMemcpyDeviceToHost, streams[i]) );
    }


    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaDeviceSynchronize() ); 
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        total_tc_final += total_triangle_h[i]; 
    }

    printf("Total number of triangle: %d\n", total_tc_final); 

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaFree(input_row_ind_d[i]) );
        CUDA_RT_CALL( cudaFree(input_col_ind_d[i]) );
        CUDA_RT_CALL( cudaFree(input_csr_row_d[i]) );
        CUDA_RT_CALL( cudaFree(total_triangle_d[i]) );
        CUDA_RT_CALL( cudaStreamDestroy(streams[i]) ); 
    }

    free(input_row_ind_h);
    free(input_col_ind_h);
    free(input_csr_row_h);
    free(total_triangle_h);
}