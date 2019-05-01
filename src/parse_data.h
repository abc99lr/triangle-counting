#include <algorithm>
#include <cstdlib>
#include <stdio.h>
#include <string>

// change this to the path of graph file



 // src_vec: the vector of src points
 // dst_vec: the vector of dst points
int read_tsv(int* src_vec, int* dst_vec, int* csr_row, int row_size, int vec_size, char* file) {
  FILE* fp_;
  fp_ = fopen(file, "r");
  if (fp_ == NULL){
    printf("No file %s found. \n", file);
    exit(-1);
  }
  // printf("Parsing the TSV file...\n");

  int pos = 0;

  while (true) {
    // if (pos % 100000 == 0){
      // printf("Finished parsing %d edges.\n", pos);
    // }
    int dst, src, weight;
    const size_t numFilled = fscanf(fp_, "%d %d %d", &dst, &src, &weight);
    if (numFilled != 3) {
      break;
    }

    if(src > dst){
      src_vec[pos] = src-1;
      dst_vec[pos] = dst-1;
      pos++;
    }
  }
  // convert to csr
  int i =0;
  csr_row[0] = 0;
  int row_val = 0;
  int count = 0;
  int row_ptr = 1;
  while(i < vec_size){
    if(row_val == src_vec[i]){
      count++;
      i++;
    }else if(row_val == src_vec[i]-1){
      csr_row[row_ptr] = csr_row[row_ptr-1] + count;
      row_ptr++;
      count =1;
      row_val = src_vec[i];
      i++;
    }else{
      csr_row[row_ptr] = csr_row[row_ptr-1];
      row_ptr++;
      count =0;
      row_val++;
    }
  }
  int end = csr_row[row_ptr-1] + 1;
  while(row_val < row_size){
    csr_row[row_ptr] = end;
    row_ptr++;
    row_val++;
  }
  return pos;
}



