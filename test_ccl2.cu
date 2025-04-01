#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <fstream>

// Constants
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 4
#define WARP_SIZE 32

// Distance operators
__device__ unsigned start_distance(unsigned pixels, unsigned tx) {
    return __clz(~(pixels << (32-tx)));
}

__device__ unsigned end_distance(unsigned pixels, unsigned tx) {
    return __ffs(~(pixels >> (tx+1)));
}

// Build bitmask of pixels in a warp
__device__ unsigned buildBitmask(int value, int laneId) {
    // Use ballot_sync to create a bitmask where each bit represents a thread's predicate
    return __ballot_sync(0xFFFFFFFF, value);
}

// Find representative label using path compression
__device__ unsigned find(unsigned* labels, unsigned x) {
    unsigned y = x;
    while (y != labels[y]) {
        y = labels[y];
    }
    
    // Path compression - update all nodes in the path to point to the root
    while (x != y) {
        unsigned tmp = labels[x];
        labels[x] = y;
        x = tmp;
    }
    
    return y;
}

// Union operation for union-find algorithm
__device__ void unionLabels(unsigned* labels, unsigned a, unsigned b) {
    while ((a != b) && a != labels[a]) {
        a = labels[a];
    }
    
    while ((a != b) && b != labels[b]) {
        b = labels[b];
    }

    while (a != b) {
        if (a < b) {
            unsigned tmp = a;
            a = b;
            b = tmp;
        }
        unsigned c = atomicMin(&labels[a], b);
        if (a == c) {
            a = b;
        } else {
            a = c;
        }
    }
}

std::vector<bool> loadPGM(const std::string& filename, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return {};
    }

    std::string magic;
    file >> magic;
    if (magic != "P5") {
        std::cerr << "Error: Invalid PGM file format." << std::endl;
        return {};
    }

    file >> width >> height;
    int max_val;
    file >> max_val;

    file.ignore(); // Ignore newline or carriage return before pixel data

    std::vector<unsigned char> pixel_data(width * height);
    file.read(reinterpret_cast<char*>(pixel_data.data()), pixel_data.size());

    // Convert to vector<bool>
    std::vector<bool> pixels(width * height);
    for (size_t i = 0; i < pixel_data.size(); ++i) {
        pixels[i] = (pixel_data[i] > 0); // Set to true if pixel value > 0
    }

    file.close();

    return pixels;
}

//------------------------------------------------------------------------------
// Kernel 1: Strip Labeling
//------------------------------------------------------------------------------
__global__ void stripLabelingMerging(const bool* input, unsigned* labels, int width, int height, int pitch)
{
    __shared__ unsigned bitmasks[BLOCK_HEIGHT];

    int blockStartRow = blockIdx.y * BLOCK_HEIGHT;
    int row = blockStartRow + threadIdx.y;
    int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;                                             // ky,x
    
    if (row >= height || col >= width) return;
    
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    
    bool pixelValue = (col < width && row < height) ? input[row * pitch + col] > 0 : 0;           // py,x
    unsigned warpBitmask = __ballot_sync(0xFFFFFFFF, pixelValue);                                 // pixels_y
    unsigned startDist = start_distance(warpBitmask, laneId);                                     // s_dist_y

    if (pixelValue && (startDist == 0)) {
        int idx = row * width + col;
        labels[idx] = idx;
    }

    if (laneId == 0) {
        bitmasks[warpId] = warpBitmask;
    }
    __syncthreads();

    if (warpId > 0) {
        bool pixelValue_last_y = input[(row - 1) * pitch + col] > 0;                             // py-1,x
        unsigned warpBitmask_last_y = bitmasks[warpId-1];                                        // pixels_y-1
        unsigned startDist_last_y = start_distance(warpBitmask_last_y, laneId);                  // s_dist_y-1
        
        if (pixelValue && pixelValue_last_y && (startDist == 0 || startDist_last_y == 0)) {
            unsigned label1 = row * width + col - startDist;
            unsigned label2 = (row-1) * width + col - startDist_last_y;
            unionLabels(labels, label1, label2);
        }
    }
    __syncthreads();

    if (blockIdx.y == 0) {
        for (int blk = 1; blk < height/BLOCK_HEIGHT; ++blk) {
            int row = blk * BLOCK_HEIGHT + threadIdx.y;
            int col = threadIdx.x;

            int ky = row * width + col;
            int k_last_y = (row - 1) * width + col;

            bool py = input[ky];
            bool p_last_y = input[k_last_y];

            unsigned warpBitmask = __ballot_sync(0xFFFFFFFF, py);
            unsigned warpBitmask_last_y = __ballot_sync(0xFFFFFFFF, p_last_y);

            if (py && p_last_y) {
                unsigned startDist = start_distance(warpBitmask, laneId);
                unsigned startDist_last_y = start_distance(warpBitmask_last_y, laneId);
    
                if (startDist == 0 || startDist_last_y == 0) {
                    int global_label1 = ky - startDist;
                    int global_label2 = k_last_y - startDist_last_y;
    
                    unionLabels(labels, global_label1, global_label2);
                    
                }
                
            }
        }
    }

}

__global__ void stripRelabeling(const bool* input, unsigned* labels, int width, int height, int pitch) {

    int row = blockIdx.y * BLOCK_HEIGHT + threadIdx.y;
    int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;                                             // ky,x
    
    if (row >= height || col >= width) return;
    
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;

    // relabel
    int ky = row * width + col;
    bool py = input[ky];
    unsigned pixel = __ballot_sync(0xFFFFFFFF, py);
    unsigned s_dist = start_distance(pixel, laneId);
    int label = 0;

    if (py && (s_dist == 0)) {
        label = labels[ky];

        while (label != labels[label]) {
            label = labels[label];
        }
    }

    label = __shfl_sync(0xFFFFFFFF, label, laneId - s_dist);
    if (py) labels[ky] = label;
}

void previewImage(const std::vector<bool>& image, int width, int height) {
    std::cout << "Image Preview:\n";
    
    int step_y = std::max(1, height / 64);
    int step_x = std::max(1, width / 64);

    for (int row = 0; row < height; row += step_y) {
        for (int col = 0; col < width; col += step_x) {
            char pixel = (image[row * width + col] > 0) ? '#' : ' ';
            std::cout << pixel;
        }
        std::cout << '\n';
    }
}

/*int main() {

    int width, height;
    std::vector<bool> h_input = loadPGM("test_input.pgm", width, height);
    if (h_input.empty()) {
        return 1; // Error loading file
    }

    previewImage(h_input, width, height);

    // Allocate device memory and copy input data
    thrust::device_vector<bool> d_input = h_input;
    thrust::device_vector<unsigned> d_labels(width * height, 0);

    thrust::device_vector<unsigned> d_output(width * height);

    // Launch CUDA kernel
    dim3 blocks((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);
    dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);

    std::cout<<"blocks size: "<<blocks.x<<", "<<blocks.y<<std::endl;
    std::cout<<"threads size: "<<threads.x<<", "<<threads.y<<std::endl;

    
    
    stripLabelingMerging<<<blocks, threads>>>(thrust::raw_pointer_cast(d_input.data()),
                                       thrust::raw_pointer_cast(d_labels.data()),
                                       width, height, width);

    stripRelabeling<<<blocks, threads>>>(thrust::raw_pointer_cast(d_input.data()),
                                              thrust::raw_pointer_cast(d_labels.data()),
                                              width, height, width);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    thrust::host_vector<unsigned> h_labels = d_labels;

    // Allocate memory for component features (assuming max components = width*height/4)
    // Each component stores: [area, min_x, max_x, min_y, max_y]
    thrust::device_vector<unsigned> d_features(width * height * 5, 0);
    thrust::host_vector<unsigned> h_output = d_output;
    

    for (int i=0;i<height;++i) {
        for (int j=0;j<width;++j) {
            std::cout<<h_labels[i*width + j]<<", ";
        }
        std::cout<<std::endl;
    }
    
    
    return 0;
}*/