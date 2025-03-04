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
__device__ unsigned buildBitmask(int value, int tid) {
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
    unsigned rootA = find(labels, a);
    unsigned rootB = find(labels, b);
    
    if (rootA != rootB) {
        // Make the smaller label the representative
        if (rootA < rootB) {
            labels[rootB] = rootA;
        } else {
            labels[rootA] = rootB;
        }
    }
}

std::vector<unsigned char> loadPGM(const std::string& filename, int& width, int& height) {
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

    std::vector<unsigned char> pixels(width * height);
    file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());

    file.close();

    return pixels;
}

//------------------------------------------------------------------------------
// Kernel 1: Strip Labeling
//------------------------------------------------------------------------------
__global__ void stripLabeling(
    const unsigned char* input,
    unsigned* labels,
    int width,
    int height,
    int pitch)
{
    // Each block processes a horizontal strip of 4 rows
    int blockStartRow = blockIdx.y * BLOCK_HEIGHT;
    int row = blockStartRow + threadIdx.y;
    int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    
    // Out of bounds check
    if (row >= height || col >= width) {
        return;
    }
    
    // Thread ID within the warp
    int tid = threadIdx.x;
    int warpId = threadIdx.y;
    
    // Shared memory for the bitmasks and temporary labels
    __shared__ unsigned bitmasks[BLOCK_HEIGHT];
    __shared__ unsigned temp_labels[BLOCK_HEIGHT][BLOCK_WIDTH];
    
    // Get the pixel value (0 or 1)
    unsigned char pixelValue = (col < width && row < height) ? input[row * pitch + col] > 0 : 0;
    
    // Build the bitmask for the current warp
    unsigned warpBitmask = buildBitmask(pixelValue, tid);
    
    // Store the bitmask in shared memory (only the first thread in each warp)
    if (tid == 0) {
        bitmasks[warpId] = warpBitmask;
    }
    
    __syncthreads();
    
    // Process only set pixels
    if (pixelValue) {
        unsigned startDist = start_distance(bitmasks[warpId], tid);
        
        // If this pixel is the start of a segment (startDist == 0)
        if (startDist == 0) {
            // Assign a new label based on global position
            unsigned label = row * width + col;
            temp_labels[warpId][tid] = label;
            
            // Initialize the label in the union-find data structure
            labels[label] = label;
        } else {
            // This pixel is part of an existing segment
            // Get the label from the start pixel of the segment
            temp_labels[warpId][tid] = temp_labels[warpId][tid - startDist];
        }
    } else {
        // For non-set pixels, assign invalid label
        temp_labels[warpId][tid] = UINT_MAX;
    }
    
    __syncthreads();
    
    // Write the labels to global memory for set pixels
    if (pixelValue) {
        labels[row * width + col] = temp_labels[warpId][tid];
    }
}

void previewImage(const std::vector<unsigned char>& image, int width, int height) {
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

int main() {

    int width, height;
    std::vector<unsigned char> h_input = loadPGM("test_input.pgm", width, height);
    if (h_input.empty()) {
        return 1; // Error loading file
    }

    previewImage(h_input, width, height);

    // Allocate device memory and copy input data
    thrust::device_vector<unsigned char> d_input = h_input;
    thrust::device_vector<unsigned> d_labels(width * height, 0);

    // Launch CUDA kernel
    dim3 blocks((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);
    dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);

    std::cout<<"blocks size: "<<blocks.x<<", "<<blocks.y<<std::endl;
    std::cout<<"threads size: "<<threads.x<<", "<<threads.y<<std::endl;

    stripLabeling<<<blocks, threads>>>(thrust::raw_pointer_cast(d_input.data()),
                                       thrust::raw_pointer_cast(d_labels.data()),
                                       width, height, width);

    thrust::host_vector<unsigned> h_labels = d_labels;
    
    
    return 0;
}