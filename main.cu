#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda.h>
#include <chrono>

using namespace std;

#define cudaErrorCheck(result) { cudaAssert((result), __FILE__, __FUNCTION__, __LINE__); }

inline void cudaAssert(cudaError_t err, const char *file,  const char *function, int line, bool quit=true) {
   if (err != cudaSuccess) {
      fprintf(stderr,"cudaAssert failed with error \'%s\', in File: %s, Function: %s, at Line: %d\n", cudaGetErrorString(err), file, function, line);
      if (quit) exit(err);
   }
}

__global__ void tiled_image_process(int mask_nrows, int mask_ncols, int img_rows, int img_cols, int tile_rows, int tile_cols, unsigned char *img, unsigned char *img_out, double *mask) {
    extern __shared__ double shared_mem[];
    double *mask_shared = shared_mem; // (double *) comes first to ensure memory alignment regardless of block size
    unsigned char *pixel_block = (unsigned char *)&shared_mem[mask_nrows*mask_ncols];
    // double *mask_shared = (double *)&pixel_block[blockDim.x*blockDim.y*3];

    int half_width = mask_ncols / 2;
    int half_height = mask_nrows / 2;
    double val_b = 0, val_g = 0, val_r = 0;

    /**
     * Tiling strategy:
     * 
     * 1. Each thread also maps to a specific input pixel (or out of bounds), which it must bring into shared memory
     * 2. Each thread maps to a specific output pixel whose final value it computes, which is mask_width/2 greater than the input
     *    pixel (arbitrary, but made for convenient indexing)
     * 3. Threads with indexes lower than mask dimensions bring the mask into shared memory as well
     * 4. Then, compute convolution using pixels and mask in shared memory
    */
    
    // pixel in output image that this thread corresponds to
    int pix_out_x = threadIdx.x + blockIdx.x*tile_cols;
    int pix_out_y = threadIdx.y + blockIdx.y*tile_rows;

    // pixel in input image that this thread needs to bring into shared mem
    int pix_in_x = pix_out_x - half_width;
    int pix_in_y = pix_out_y - half_height;

    // bring pixel into shared memory (small amount of thread divergence)
    if (pix_in_x >= 0 && pix_in_x < img_cols && pix_in_y >= 0 && pix_in_y < img_rows) {
        pixel_block[(threadIdx.y*blockDim.x + threadIdx.x)*3] = img[(pix_in_y*img_cols + pix_in_x)*3]; // B
        pixel_block[(threadIdx.y*blockDim.x + threadIdx.x)*3 + 1] = img[(pix_in_y*img_cols + pix_in_x)*3 + 1]; // G
        pixel_block[(threadIdx.y*blockDim.x + threadIdx.x)*3 + 2] = img[(pix_in_y*img_cols + pix_in_x)*3 + 2]; // R
    }
    if (pix_in_x < 0 || pix_in_x >= img_cols || pix_in_y < 0 || pix_in_y >= img_rows) {
        pixel_block[(threadIdx.y*blockDim.x + threadIdx.x)*3] = 0;
        pixel_block[(threadIdx.y*blockDim.x + threadIdx.x)*3 + 1] = 0;
        pixel_block[(threadIdx.y*blockDim.x + threadIdx.x)*3 + 2] = 0;
    }

    if (threadIdx.x < mask_ncols && threadIdx.y < mask_nrows) {
        mask_shared[threadIdx.y*mask_ncols + threadIdx.x] = mask[threadIdx.y*mask_ncols + threadIdx.x];
    }

    __syncthreads();

    if (threadIdx.x < tile_cols && threadIdx.y < tile_rows && pix_out_x < img_cols && pix_out_y < img_rows) {
        int pix_local_x = threadIdx.x + half_width;
        int pix_local_y = threadIdx.y + half_height;
        for (int i = -half_height; i <= half_height; i++) {
            int mask_row = i + half_height; // assuming odd-dimensioned masks
            int curr_row = pix_local_y + i;
            for (int j = -half_width; j <= half_width; j++) {
                int mask_col = j + half_width;
                int curr_col = pix_local_x + j;
                if ((curr_row >= 0 && curr_row < img_rows) && (curr_col >= 0 && curr_col < img_cols)) {
                    int curr_idx = (curr_row*blockDim.x + curr_col) * 3;
                    double mask_val = mask_shared[mask_row*mask_ncols + mask_col];
                    val_b += pixel_block[curr_idx] * mask_val;
                    val_g += pixel_block[curr_idx+1] * mask_val;
                    val_r += pixel_block[curr_idx+2] * mask_val;
                }
            }
        }
        // update output image, account for BGR indexes per pixel
        img_out[(pix_out_y*img_cols + pix_out_x)*3] = (int)floor(val_b);
        img_out[(pix_out_y*img_cols + pix_out_x)*3 + 1] = (int)floor(val_g);
        img_out[(pix_out_y*img_cols + pix_out_x)*3 + 2] = (int)floor(val_r);
    }
}

__global__ void globalmem_image_process(int mask_nrows, int mask_ncols, int img_rows, int img_cols, unsigned char *img, unsigned char *img_out, double *mask) {
    int pix_x = blockDim.x * blockIdx.x + threadIdx.x;
    int pix_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (pix_y < img_rows && pix_x < img_cols) {
        int half_height = mask_nrows / 2;
        int half_width = mask_ncols / 2;
        double val_b = 0, val_g = 0, val_r = 0;
        for (int i = -half_height; i <= half_height; i++) {
            int mask_row = i + half_height;
            int curr_row = pix_y + i;
            for (int j = -half_width; j <= half_width; j++) {
                int mask_col = j + half_width;
                int curr_col = pix_x + j;
                if ((curr_row >= 0 && curr_row < img_rows) && (curr_col >= 0 && curr_col < img_cols)) {
                    int curr_idx = (curr_row*img_cols + curr_col) * 3;
                    double mask_val = mask[mask_row*mask_ncols + mask_col];
                    val_b += img[curr_idx] * mask_val;
                    val_g += img[curr_idx+1] * mask_val;
                    val_r += img[curr_idx+2] * mask_val;
                }
            }
        }
        img_out[(pix_y*img_cols + pix_x)*3] = (int)floor(val_b);
        img_out[(pix_y*img_cols + pix_x)*3 + 1] = (int)floor(val_g);
        img_out[(pix_y*img_cols + pix_x)*3 + 2] = (int)floor(val_r);
    }
}

/** serial image processing for GPU speedup reference */
void serial_img_process(const cv::Mat& image, cv::Mat& processed_image, const vector<double> &mask, int mask_nrows, int mask_ncols) {
    int halfheight = mask_nrows/2;
    int halfwidth = mask_ncols/2;

    for (int pix_y = 0; pix_y < image.rows; pix_y++)
    {
        for (int pix_x = 0; pix_x < image.cols; pix_x++)
        {
            double val_b = 0, val_g = 0, val_r = 0;
            for (int i = -halfheight; i <= halfheight; i++) {
                int mask_row = i + halfheight;
                int curr_row = pix_y + i;
                for (int j = -halfwidth; j <= halfwidth; j++) {
                    int mask_col = j + halfwidth;
                    int curr_col = pix_x + j;
                    if ((curr_row >= 0 && curr_row < image.rows) && (curr_col >=0 && curr_col < image.cols)) {
                        val_b += image.at<cv::Vec3b>(curr_row, curr_col)[0] * mask[mask_row*mask_ncols + mask_col];
                        val_g += image.at<cv::Vec3b>(curr_row, curr_col)[1] * mask[mask_row*mask_ncols + mask_col];
                        val_r += image.at<cv::Vec3b>(curr_row, curr_col)[2] * mask[mask_row*mask_ncols + mask_col];
                    }
                }
            }
            processed_image.at<cv::Vec3b>(pix_y, pix_x)[0] = (int)floor(val_b);
            processed_image.at<cv::Vec3b>(pix_y, pix_x)[1] = (int)floor(val_g);
            processed_image.at<cv::Vec3b>(pix_y, pix_x)[2] = (int)floor(val_r);
        }
    }
}

vector<double> parse_mask(const string &fname, int &nrows, int &ncols) {
    ifstream file(fname);
    if (!file.is_open()) {
        cerr << "Failed to open kernel/mask file." << endl;
        exit(1);
    }
    string line;
    vector<double> mask;
    nrows = ncols = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            double value = stod(cell);
            mask.push_back(value);
            if (!nrows) ncols++;
        }
        nrows++;
    }
    file.close();
    return mask;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Input format: <image-filename> <mask-filename> <output-image-filename> <block_x> <block_y>" << endl;
        exit(1);
    }

    string imagefile = argv[1];
    string maskfile = argv[2];
    string outputfile = argv[3];
    int blocksize_x = atoi(argv[4]);
    int blocksize_y = atoi(argv[5]);
    
    cv::Mat unproc_img = cv::imread(imagefile, cv::IMREAD_COLOR);
    
    if(!unproc_img.data) {
        cerr << "Error opening image file" << endl;
        exit(1);
    }

    int mask_nrows, mask_ncols;
    vector<double> mask = parse_mask(maskfile, mask_nrows, mask_ncols);
    
    assert(mask_nrows % 2 != 0);
    assert(mask_ncols % 2 != 0);
    assert(blocksize_x > mask_ncols);
    assert(blocksize_y > mask_nrows);

    uchar *dev_img, *dev_img_out;
    double *dev_mask;

    // allocate memory and copy data to gpu
    cudaErrorCheck(cudaMalloc((void**)&dev_img, unproc_img.rows*unproc_img.cols*sizeof(uchar)*3)); // img stored contiguously in row-major, BGR entries per pixel
    cudaErrorCheck(cudaMalloc((void**)&dev_img_out, unproc_img.rows*unproc_img.cols*sizeof(uchar)*3));
    cudaErrorCheck(cudaMalloc((void**)&dev_mask, mask_nrows*mask_ncols*sizeof(double)));
    cudaErrorCheck(cudaMemcpy(dev_img, &unproc_img.at<uchar>(0,0), unproc_img.rows*unproc_img.cols*sizeof(uchar)*3, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(dev_mask, &mask[0], mask_nrows*mask_ncols*sizeof(double), cudaMemcpyHostToDevice));

    int tilesize_x = blocksize_x - 2*(mask_ncols/2);
    int tilesize_y = blocksize_y - 2*(mask_nrows/2);

    int gridx = (int)ceil((double)unproc_img.cols / (double)tilesize_x);
    int gridy = (int)ceil((double)unproc_img.rows / (double)tilesize_y);
    dim3 block_size(blocksize_x, blocksize_y);
    dim3 grid_size(gridx, gridy);
   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // run naive global memory implementation
    cudaEventRecord(start);
    globalmem_image_process<<<grid_size, block_size>>>(mask_nrows, mask_ncols, unproc_img.rows, unproc_img.cols, dev_img, dev_img_out, dev_mask);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaErrorCheck(cudaPeekAtLastError());

    float globalmem_time;
    cudaEventElapsedTime(&globalmem_time, start, stop); // ms
    globalmem_time /= 1000.0;

    cv::Mat globalmem_img(unproc_img.rows, unproc_img.cols, unproc_img.type());
    cudaMemcpy(&globalmem_img.at<uchar>(0,0), dev_img_out, globalmem_img.rows*globalmem_img.cols*sizeof(uchar)*3, cudaMemcpyDeviceToHost);
 
    // run tiled (shared memory) implementation
    cudaEventRecord(start);
    int shared_mem = blocksize_x*blocksize_y*sizeof(uchar)*3 + mask_nrows*mask_ncols*sizeof(double); // tile+halo and mask
    cout << "Shared: " << shared_mem << endl;
    tiled_image_process<<<grid_size, block_size, shared_mem>>>(mask_nrows, mask_ncols, unproc_img.rows, unproc_img.cols, tilesize_y, tilesize_x, dev_img, dev_img_out, dev_mask);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaErrorCheck(cudaPeekAtLastError());

    float tiled_time;
    cudaEventElapsedTime(&tiled_time, start, stop); // ms
    tiled_time /= 1000.0;

    cv::Mat tiled_img(unproc_img.rows, unproc_img.cols, unproc_img.type());
    cudaMemcpy(&tiled_img.at<uchar>(0,0), dev_img_out, tiled_img.rows*tiled_img.cols*sizeof(uchar)*3, cudaMemcpyDeviceToHost);
    

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // run serial implementation
    cv::Mat serial_img(unproc_img.rows, unproc_img.cols, unproc_img.type());
    auto serial_start = chrono::high_resolution_clock::now();
    serial_img_process(unproc_img, serial_img, mask, mask_nrows, mask_ncols);
    auto serial_stop = chrono::high_resolution_clock::now();
    std::chrono::duration<double> serial_time = serial_stop - serial_start;
    
    cv::imshow("Unprocessed Image", unproc_img);
    cv::imshow("Shared Memory Processed Image", tiled_img);
    cv::imshow("Global Memory Image", globalmem_img);
    cv::imshow("Serially Processed Image", serial_img);
    cv::waitKey(0);

    cout << "Shared memory parallel algorithm took: " << tiled_time << " seconds" << endl;
    cout << "Global memory parallel algorithm took: " << globalmem_time << " seconds" << endl;
    cout << "Serial algorithm took: " << serial_time.count() << " seconds" << endl;
    cout << "Shared memory vs serial speedup: " << serial_time.count() / tiled_time << endl;
    cout << "Shared memory vs global memory speedup: " << globalmem_time / tiled_time << endl;

    // verify shared memory and global memory processing gave same result
    // opencv L1 norm takes sum of absolute values of elementwise subtractions
    bool imgs_equal = !cv::norm(tiled_img, globalmem_img, cv::NORM_L1);
    cout << "Images are equal: " << (imgs_equal ? "True" : "False") << endl;
    
    // the following equality test does not work because of floating point arithmetic differences between the
    // GPU and CPU. After inspection, about ~1.5% of the pixels had either their R, G, or B values (or in rare 
    // cases multiple of them) differ by a value of 1. This is visually unnoticeable but numerically different
    
    // imgs_equal = !cv::norm(tiled_img, serial_img, cv::NORM_L1);
    // cout << "Images from Serial and GPU processing are equal: " << (imgs_equal ? "True" : "False") << endl;

    cudaFree(dev_img);
    cudaFree(dev_img_out);
    cudaFree(dev_mask);
}