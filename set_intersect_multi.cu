#include <thrust/reduce.h>
#include <sys/time.h>

#include "src/parse_data.h"

#define NUM_GPUS 2
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


__global__ static void set_intersect_multi(int *__restrict__ triangleCounts, int* edgeSrc, 
                                    int* edgeDst, int* rowPtr, int numEdges, int startIdx) {

    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < numEdges){

        int u = edgeSrc[t];
        int v = edgeDst[t + startIdx];

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
    int *output_d[NUM_GPUS];
    cudaStream_t streams[NUM_GPUS];

    int *input_row_ind_h, *input_col_ind_h, *input_csr_row_h;
    int *output_h; 

    struct timeval t1, t2;
    double mytime;

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
    output_h = (int*) malloc((num_edges * sizeof(int)));

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

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 

        CUDA_RT_CALL( cudaMalloc((void **)&input_row_ind_d[i], num_edges_each[i] * sizeof(int)) );
        CUDA_RT_CALL( cudaMalloc((void **)&input_col_ind_d[i], num_edges * sizeof(int)) );
        CUDA_RT_CALL( cudaMalloc((void **)&input_csr_row_d[i], (num_nodes + 1) * sizeof(int)) );
        CUDA_RT_CALL( cudaMalloc((void **)&output_d[i], num_edges_each[i] * sizeof(int)) );
        CUDA_RT_CALL( cudaMemset(output_d[i], 0, num_edges_each[i] * sizeof(int)) );
        CUDA_RT_CALL( cudaStreamCreate(&streams[i]) );
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaMemcpyAsync(input_row_ind_d[i], input_row_ind_h + i * length_common, num_edges_each[i] * sizeof(int), cudaMemcpyHostToDevice, streams[i]) );
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
        dim3 dimGrid(ceil(num_edges_each[i] / 128.0));
        set_intersect_multi<<<dimGrid, dimBlock, 0, streams[i]>>>(output_d[i], input_row_ind_d[i], input_col_ind_d[i], input_csr_row_d[i], num_edges_each[i], i * length_common);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaDeviceSynchronize() ); 
    } 

    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in real work: %3.5f ms\n", mytime); 
    
    int total_tc[NUM_GPUS]; 
    int total_tc_final = 0;
    gettimeofday(&t1, 0);
    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        total_tc[i] = thrust::reduce(thrust::device, output_d[i], output_d[i] + num_edges_each[i]);
    }
    

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaDeviceSynchronize() ); 
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        total_tc_final += total_tc[i]; 
    }

    gettimeofday(&t2, 0);
    mytime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Time elapsed in reduce: %3.5f ms\n", mytime); 
    printf("Total number of triangle: %d\n", total_tc_final); 

    for (int i = 0; i < NUM_GPUS; i++) {
        CUDA_RT_CALL( cudaSetDevice(gpu_ids[i]) ); 
        CUDA_RT_CALL( cudaFree(input_row_ind_d[i]) );
        CUDA_RT_CALL( cudaFree(input_col_ind_d[i]) );
        CUDA_RT_CALL( cudaFree(input_csr_row_d[i]) );
        CUDA_RT_CALL( cudaFree(output_d[i]) );
        CUDA_RT_CALL( cudaStreamDestroy(streams[i]) ); 
    }

    free(input_row_ind_h);
    free(input_col_ind_h);
    free(input_csr_row_h);
    free(output_h);
}