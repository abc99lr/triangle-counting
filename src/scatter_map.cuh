#define BLOCK_SIZE 256
// #define SIZE_OF_INT 32

__global__ void scatter_with_map(int num_edges, int num_nodes, int *input_row_ind, int *input_col_ind, int *input_csr_row, char *byte_map, int *total_triangle) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < num_edges) {
        int row = input_row_ind[tx];
        int internal = input_col_ind[tx];
        #pragma unroll
        for (int i = input_csr_row[internal]; i < input_csr_row[internal + 1]; i++) {
            int col = input_col_ind[i]; 
            int index = row * num_nodes + col; 
            // int entry = index / SIZE_OF_INT; 
            // int offset = index % SIZE_OF_INT; 

            // if ((bit_map[entry] >> offset) & 1 == 1) {
            if (byte_map[index] == 1) {
                atomicAdd(total_triangle, 1); 
            }
        }
    }
}

__global__ void build_bit_map(int num_edges, int num_nodes, int *input_row_ind, int *input_col_ind, char *byte_map) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < num_edges) {
        int index = input_row_ind[tx] * num_nodes + input_col_ind[tx]; 
        // int entry = index / SIZE_OF_INT; 
        // int offset = index % SIZE_OF_INT; 

        // int to_add = 1 << offset; 
        // atomicAdd(&bit_map[entry], to_add); 
        // bit_map[entry] |= 1 << offset;
        byte_map[index] = 1; 
    }
}