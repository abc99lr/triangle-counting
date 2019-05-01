gather_reduce : gather_reduce.cu
	nvcc --std=c++11 -Ithrust -I. -Xcompiler -fopenmp gather_reduce.cu -o gather_reduce

gather_atomic : gather_atomic.cu
	nvcc --std=c++11 -I. -Xcompiler -fopenmp gather_atomic.cu -o gather_atomic

scatter_map : scatter_map.cu
	nvcc --std=c++11 -I. -Xcompiler -fopenmp scatter_map.cu -o scatter_map

scatter_hash : scatter_hash.cu
	nvcc --std=c++11 -I. -Ilibgdf -Xcompiler -fopenmp scatter_hash.cu -o scatter_hash

set_intersect : set_intersect.cu
	nvcc -Ithrust -I. --std=c++11 set_intersect.cu -o set_intersect

set_multi : set_intersect_multi.cu
	nvcc -Ithrust -I. --std=c++11 set_intersect_multi.cu -o set_multi

spgemm : spgemm.cu
	nvcc -Ithrust -Icudf/cpp/src/ -Icudf/cpp/include/ -Icudf/cpp/thirdparty/rmm/include/ -I. --std=c++11 spgemm.cu -o spgemm

clean:
	rm -rf spgemm set_intersect scatter_map scatter_hash gather_reduce gather_atomic set_multi
