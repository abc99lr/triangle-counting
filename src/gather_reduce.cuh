#define BLOCK_SIZE 256

__global__ void gather_naive(int input_size, int *input_row_ind, int *input_col_ind, int *input_csr_row, int *output_value) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < input_size) {
        int row = input_row_ind[tx];
        int col = input_col_ind[tx];

        int sum = 0; 
        #pragma unroll
        for (int i = input_csr_row[row]; i < input_csr_row[row + 1]; i++) {
            int interal = input_col_ind[i]; 
            #pragma unroll
            for (int j = input_csr_row[interal]; j < input_csr_row[interal + 1]; j++) {
                
                if (input_col_ind[j] == col) {
                    sum++; 
                }
            }
        }
        
        output_value[tx] = sum; 
    }
}

void make_col_index(int *dest_vec, int *col_index, int col_size, int vec_size){
    int i =0;
    col_index[0] = 0;
    int col_val = 0;
    int count = 0;
    int col_ptr = 1;
    while(i < vec_size){
        if(col_val == dest_vec[i]){
            count++;
            i++;
        }else if(col_val == dest_vec[i]-1){
            col_index[col_ptr] = col_index[col_ptr-1] + count;
            col_ptr++;
            count =1;
            col_val = dest_vec[i];
            i++;
        }else{
            col_index[col_ptr] = col_index[col_ptr-1];
            col_ptr++;
            count =0;
            col_val++;
        }
    }
    int end = col_index[col_ptr-1] + 1; 
    while(col_val < col_size){
        col_index[col_ptr] = end;
        col_ptr++;
        col_val++;
    }
}

__global__ void gather_binned(int input_size, int *input_row_ind, int *input_col_ind, int *input_csr_row,
                            int *input_cm_row_ind, int *input_cm_col_ind, int *input_cm_col, int *output_value){
    int t = threadIdx.x + blockIdx.x*blockDim.x;
    if(t < input_size){
        int row = input_row_ind[t];
        int col = input_col_ind[t];
        int i = input_csr_row[row];
        int i_end = input_csr_row[row+1];
        int j = input_cm_col[col];
        int j_end = input_cm_col[col+1];

        int w1 = input_col_ind[i];
        int w2 = input_cm_row_ind[j];
        int sum = 0;
        while(i < i_end && j < j_end){
            if(w1 < w2){
                w1 = input_col_ind[++i];
            }else if(w2 < w1){
                w2 = input_cm_row_ind[++j];
            }else{
                w1 = input_col_ind[++i];
                w2 = input_cm_row_ind[++j];
                sum++;
            }
        }
        output_value[t] = sum;
    }
}
