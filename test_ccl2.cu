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
__global__ void stripLabeling(
    const bool* input,
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
            unsigned startLabel = temp_labels[warpId][tid - startDist];
            temp_labels[warpId][tid] = startLabel;
            
            // Merge current pixel's label with the start pixel's label
            unsigned currentLabel = row * width + col;
            unionLabels(labels, startLabel, currentLabel);
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

//------------------------------------------------------------------------------
// Kernel 2: Border Merging
//------------------------------------------------------------------------------
__global__ void borderMerging(
    const bool* input,    // 輸入二進制圖像
    unsigned* labels,     // 標籤數組
    int width,            // 圖像寬度
    int height,           // 圖像高度
    int pitch             // 圖像pitch
) {
    // 使用二維網格和塊
    int y = blockIdx.y;   // 當前條帶的 y 座標
    
    // 只處理非第一個條帶的邊界
    if (y == 0) return;
    
    // 計算上一個條帶底部的行
    int last_row_bottom = y * blockDim.y * BLOCK_HEIGHT - 1;
    
    // 確保不超出圖像高度
    if (last_row_bottom >= height - 1) return;
    
    // 每個線程處理一列
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    
    // 獲取當前列上下兩行的像素值
    int kyx_prev = last_row_bottom * width + col;
    int kyx = (last_row_bottom + 1) * width + col;
    
    // 使用 __ballot_sync 檢查這兩行的像素值
    uint32_t pyx_prev = __ballot_sync(0xFFFFFFFF, input[kyx_prev] > 0);
    uint32_t pyx = __ballot_sync(0xFFFFFFFF, input[kyx] > 0);
    
    
    // 如果上下兩行都有像素
    if (pyx && pyx_prev) {
        // 計算像素的起始距離
        int s_dist_y_prev = start_distance(pyx_prev, threadIdx.x);
        int s_dist_y = start_distance(pyx, threadIdx.x);
        
        
        // 如果是段的開始
        if (s_dist_y == 0 || s_dist_y_prev == 0) {
            // 合併標籤
            printf("(%d, %d), curr: %d, next: %d\n", last_row_bottom, col, kyx - s_dist_y, kyx_prev - s_dist_y_prev);
            unionLabels(labels, kyx - s_dist_y, kyx_prev - s_dist_y_prev);
        }
    }
}

//------------------------------------------------------------------------------
// Kernel 3: Relabeling / Features Computation
//------------------------------------------------------------------------------
__global__ void relabelingAndFeatures(
    unsigned* output,
    const bool* input,
    unsigned* labels,
    unsigned* features, // Array to store component features (area, bounding box, etc.)
    int width,
    int height,
    int pitch)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= height || col >= width) {
        return;
    }
    
    // Get the pixel value
    unsigned char pixelValue = input[row * pitch + col];
    
    // Process only set pixels
    if (pixelValue > 0) {
        // Get the pixel's label
        unsigned label = labels[row * width + col];
        
        // Find the representative label
        unsigned rootLabel = find(labels, label);
        //printf("label: %d, rootLabel: %d, (%d, %d)\n", label, rootLabel, row, col);
        
        // Update the label
        labels[row * width + col] = rootLabel;
        
        // Write the final label to the output image (normalized to visible range)
        output[row * pitch + col] = (rootLabel % 255) + 1;
        
        // Update component features (example: counting area using atomic add)
        atomicAdd(&features[rootLabel * 4 + 0], 1); // Area
        
        // Update bounding box (using atomic min/max)
        atomicMin(&features[rootLabel * 4 + 1], col); // Min X
        atomicMax(&features[rootLabel * 4 + 2], col); // Max X
        atomicMin(&features[rootLabel * 4 + 3], row); // Min Y
        atomicMax(&features[rootLabel * 4 + 4], row); // Max Y
    } else {
        // Background pixels get zero
        output[row * pitch + col] = 0;
    }
}

__global__ void relabelingKernel(
    const bool* input,
    unsigned* labels, 
    int width, 
    int height)
{
    // 計算全局索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 邊界檢查，防止訪問非法記憶體
    if (x >= width || y >= height) return;
    
    // 計算 1D 索引
    unsigned ky_x = y * width + x;
    if (ky_x >= width * height) return;  // 額外檢查
    
    // 讀取 pixel 標籤
    unsigned py_x = labels[ky_x];
    
    // 確保 `py_x` 是 bool 值
    unsigned pixels = __ballot_sync(0xFFFFFFFF, py_x != 0);
    
    // 計算距離
    unsigned s_dist = start_distance(pixels, threadIdx.x);
    s_dist = min(s_dist, threadIdx.x);  // 避免負數

    unsigned label;
    
    // Relabeling 過程
    if (py_x && s_dist == 0) {
        label = labels[ky_x];
        
        // 確保 `label` 在合法範圍內
        while (label < width * height && label != labels[label]) {
            label = labels[label];
        }
    }
    
    // 保證 `__shfl_sync` 參數合法
    int src_lane = max(0, threadIdx.x - s_dist);
    label = __shfl_sync(0xFFFFFFFF, label, src_lane);
        
    // 更新 labels
    if (py_x) labels[ky_x] = label;
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

int main() {

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

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    thrust::host_vector<unsigned> h_labels = d_labels;
    
    stripLabeling<<<blocks, threads>>>(thrust::raw_pointer_cast(d_input.data()),
                                       thrust::raw_pointer_cast(d_labels.data()),
                                       width, height, width);

    
    

    // Allocate memory for component features (assuming max components = width*height/4)
    // Each component stores: [area, min_x, max_x, min_y, max_y]
    thrust::device_vector<unsigned> d_features(width * height * 5, 0);
    
    
    // Grid and block dimensions for border merging
    dim3 borderBlock(BLOCK_WIDTH, 1);
    dim3 borderGrid((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);
    
    // Grid and block dimensions for relabeling
    dim3 relabelBlock(16, 16);
    dim3 relabelGrid((width + 16 - 1) / 16, (height + 16 - 1) / 16);
    
    // Run the kernels
    
    
    borderMerging<<<borderGrid, borderBlock>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_labels.data()), width, height, width);

    
    

    /*relabelingAndFeatures<<<relabelGrid, relabelBlock>>>(
        thrust::raw_pointer_cast(d_output.data()), thrust::raw_pointer_cast(d_input.data()), 
        thrust::raw_pointer_cast(d_labels.data()),
        thrust::raw_pointer_cast(d_features.data()),
        width, height, width);*/
    
    relabelingKernel<<<relabelGrid, relabelBlock>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_labels.data()), width, height);
    
    

    thrust::host_vector<unsigned> h_output = d_output;
    

    for (int i=0;i<height;++i) {
        for (int j=0;j<width;++j) {
            std::cout<<h_labels[i*width + j]<<", ";
        }
        std::cout<<std::endl;
    }
    
    
    return 0;
}