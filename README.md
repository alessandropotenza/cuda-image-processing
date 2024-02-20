# cuda-image-processing
This project is an attempt to learn more about CUDA and GPUs in a fun way, building on concepts introduced in my Parallel Processing course (ECE 4530). It leverages GPU-parallelism to accelerate spatial image filtering, where user-provided convolution masks (defined in CSV format) are convolved with an image to produce specific effects. At first, my implementation used global memory only and achieved >600x speedup. I then explored shared memory in an attempt to minimize memory access latencies and achieve further speedup. This involved partitioning the image into tiles for distribution among blocks, creating a mapping between threads and specific input and output pixels, and ensuring that halo rows required for the convolution were accounted for in each block.

I avoided researching any tiling solutions to this task for the sake of problem solving, so there is likely a more efficient solution than mine. While I do decrease the number of global memory accesses (and increase the compute to global memory access ratio) substantially, I rarely see improved performance using my shared memory approach over my global memory one. I think this is likely due to coalesced memory accesses in the global memory implementation, which will occur because neighbouring threads access contiguous locations in memory throughout the convolution process.
