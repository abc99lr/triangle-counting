#define BLOCK_SIZE 256
#define WARP_SiZE 32
#define NUM_WARP (BLOCK_SIZE / WARP_SiZE)

__global__ void gather_atomic_naive(int input_size, int *input_row_ind, int *input_col_ind, int *input_csr_row, int *total_triangle) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < input_size) {
        int row = input_row_ind[tx];
        int col = input_col_ind[tx];
        #pragma unroll
        for (int i = input_csr_row[row]; i < input_csr_row[row + 1]; i++) {
            int interal = input_col_ind[i]; 
            #pragma unroll
            for (int j = input_csr_row[interal]; j < input_csr_row[interal + 1]; j++) {
                
                if (input_col_ind[j] == col) {
                    atomicAdd(total_triangle, 1); 
                }
            }
        }
    }
}

__global__ void gather_atomic_block(int input_size, int *input_row_ind, int *input_col_ind, int *input_csr_row, int *total_triangle) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int block_triangle;

    if (threadIdx.x == 0) { 
        block_triangle = 0; 
    }

    __syncthreads(); 

    if (tx < input_size) {
        int row = input_row_ind[tx];
        int col = input_col_ind[tx]; 
        #pragma unroll
        for (int i = input_csr_row[row]; i < input_csr_row[row + 1]; i++) {
            int interal = input_col_ind[i]; 
            #pragma unroll
            for (int j = input_csr_row[interal]; j < input_csr_row[interal + 1]; j++) {
                
                if (input_col_ind[j] == col) {
                    atomicAdd(&block_triangle, 1); 
                }
            }
        }
    }
    
    __syncthreads();  

    if (threadIdx.x == 0) {
        atomicAdd(total_triangle, block_triangle); 
    }
}

__global__ void gather_atomic_warp(int input_size, int *input_row_ind, int *input_col_ind, int *input_csr_row, int *total_triangle) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int warp_triangle[NUM_WARP]; 

    if (threadIdx.x < NUM_WARP) { 
        warp_triangle[threadIdx.x] = 0; 
    }

    __syncthreads(); 

    if (tx < input_size) {
        int row = input_row_ind[tx];
        int col = input_col_ind[tx]; 
        #pragma unroll
        for (int i = input_csr_row[row]; i < input_csr_row[row + 1]; i++) {
            int interal = input_col_ind[i]; 
            #pragma unroll
            for (int j = input_csr_row[interal]; j < input_csr_row[interal + 1]; j++) {
                
                if (input_col_ind[j] == col) {
                    atomicAdd(&warp_triangle[threadIdx.x % NUM_WARP], 1); 
                }
            }
        }
    }
    
    __syncthreads();  

    if (threadIdx.x < NUM_WARP) { 
        atomicAdd(total_triangle, warp_triangle[threadIdx.x]); 
    }
}