#define BLOCK_SIZE 256
// #define SIZE_OF_INT 32

using map_type = concurrent_unordered_map<long, int, std::numeric_limits<long>::max(), default_hash<long>, 
                                        equal_to<long>, legacy_allocator<thrust::pair<long, int>>>;
using op_type = max_op<long>; 

__global__ void scatter_with_hash_map(int num_edges, int num_nodes, int *input_row_ind, int *input_col_ind, int *input_csr_row, map_type *hash_map, int *total_triangle, int start_idx) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < num_edges) {
        long row = input_row_ind[tx];
        int internal = input_col_ind[tx + start_idx];
        #pragma unroll
        for (int i = input_csr_row[internal]; i < input_csr_row[internal + 1]; i++) {

            long col = input_col_ind[i]; 
            long index = (row << 32) | col;

            auto found = hash_map->find(index);
            if (hash_map->end() != found) {
                atomicAdd(total_triangle, 1); 
            }
        }
    }
}

// __global__ void scatter_with_hash_map_block(int num_edges, int num_nodes, int *input_row_ind, int *input_col_ind, int *input_csr_row, map_type *hash_map, int *total_triangle) {
//     int tx = blockIdx.x * blockDim.x + threadIdx.x;

//     __shared__ int block_triangle;

//     if (threadIdx.x == 0) { 
//         block_triangle = 0; 
//     }

//     __syncthreads(); 
    
//     if (tx < num_edges) {
//         int row = input_row_ind[tx];
//         int internal = input_col_ind[tx];
//         #pragma unroll
//         for (int i = input_csr_row[internal]; i < input_csr_row[internal + 1]; i++) {
//             int col = input_col_ind[i]; 
//             int index = row * num_nodes + col; 
//             // int entry = index / SIZE_OF_INT; 
//             // int offset = index % SIZE_OF_INT; 

//             // if ((bit_map[entry] >> offset) & 1 == 1) {
//             auto found = hash_map->find(index);
//                 if (hash_map->end() != found) {
//             // if (byte_map[index] == 1) {
//                 atomicAdd(&block_triangle, 1); 
//             }
//         }
//     }

//     __syncthreads();  

//     if (threadIdx.x == 0) {
//         atomicAdd(total_triangle, block_triangle); 
//     }
// }

// __global__ void build_bit_map(int num_edges, int num_nodes, int *input_row_ind, int *input_col_ind, char *byte_map) {
// 	int tx = blockIdx.x * blockDim.x + threadIdx.x;

// 	if (tx < num_edges) {
// 		int index = input_row_ind[tx] * num_nodes + input_col_ind[tx]; 
// 		// int entry = index / SIZE_OF_INT; 
// 		// int offset = index % SIZE_OF_INT; 

// 		// int to_add = 1 << offset; 
// 		// atomicAdd(&bit_map[entry], to_add); 
// 		// bit_map[entry] |= 1 << offset;
// 		byte_map[index] = 1; 
// 	}
// }

template<typename map_type>
__global__ void build_hash_map(map_type *hash_map, int *input_row_ind, int *input_col_ind, int num_edges, int num_nodes, int start_idx)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    while (tx < num_edges) {
        // long long index = input_row_ind[tx] << 32 + input_col_ind[tx]; 
        long row = input_row_ind[tx];
		long col = input_col_ind[tx + start_idx]; 
		long index = (row << 32) | col ;  
        hash_map->insert(thrust::make_pair(index, 1), max_op<long>());
        tx += blockDim.x * gridDim.x;
    }
}
