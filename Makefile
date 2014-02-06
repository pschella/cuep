all: coreas_gpu cuep

coreas_gpu: coreas_gpu.cu
	nvcc -lcudart -O2 -o coreas_gpu coreas_gpu.cu

cuep: cuep.cu
	nvcc -lcudart -O2 -o cuep cuep.cu

