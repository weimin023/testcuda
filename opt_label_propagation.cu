#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "json.hpp"

using json = nlohmann::json;


#ifndef PI
#define PI 3.14159265358979323846f
#endif

#define BACKGROUND (-1)
#define MAX_LOCAL_CHANGES (128)

#define LeftConnected (0x01)
#define UpConnected (0x02)
#define RightConnected (0x04)
#define DownConnected (0x08)
#define IsLeftConnected(x) (x & LeftConnected)
#define IsUpConnected(x) (x & UpConnected)
#define IsRightConnected(x) (x & RightConnected)
#define IsDownConnected(x) (x & DownConnected)


__device__ bool union_sets(int* parent, int x, int y) {
    auto findx = [] __device__(int* parent, int x) -> int {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    int rootX = findx(parent, x);
    int rootY = findx(parent, y);
    if (rootX == rootY) return false;

    int minRoot = (rootX < rootY) ? rootX : rootY;
    int maxRoot = (rootX < rootY) ? rootY : rootX;
    int old = atomicCAS(&parent[maxRoot], maxRoot, minRoot);
    return (old == maxRoot);
}

__global__ void label_propagation_full(int* d_labels, int* d_label_counts, const uint8_t* d_connect, int height, int width, int max_iterations,
                                       int cnt_threshold) {
    int total = height * width;
    int block_size = blockDim.x;
    int pixels_per_thread = (total + block_size - 1) / block_size;
    int start_idx = threadIdx.x * pixels_per_thread;
    int end_idx = min(start_idx + pixels_per_thread, total);

    __shared__ volatile bool changed;
    __shared__ volatile int iter;
    __shared__ volatile bool block_changed_shared;
    __shared__ bool local_changes[MAX_LOCAL_CHANGES];

    if (threadIdx.x == 0) {
        changed = true;
        iter = 0;
        block_changed_shared = false;
    }
    __syncthreads();

    auto update_pixel = [&] __device__(int idx) -> bool {
        uint8_t connect_flag = d_connect[idx];
        if (connect_flag == 0) return false;
        int row = idx / width, col = idx % width;
        bool local_updated = false;
        if ((col > 0) && IsLeftConnected(connect_flag)) local_updated |= union_sets(d_labels, idx, row * width + (col - 1));
        if ((row > 0) && IsUpConnected(connect_flag)) local_updated |= union_sets(d_labels, idx, (row - 1) * width + col);
        if ((col < (width - 1)) && IsRightConnected(connect_flag)) local_updated |= union_sets(d_labels, idx, row * width + (col + 1));
        if ((row < (height - 1)) && IsDownConnected(connect_flag)) local_updated |= union_sets(d_labels, idx, (row + 1) * width + col);
        return local_updated;
    };

    auto do_lable_propagation = [&] __device__() {
        while (changed && iter < max_iterations) {
            bool local_changed = false;
            for (int i = start_idx; i < end_idx; i++) {
                if (i < total)
                    if (update_pixel(i)) local_changed = true;
            }

            local_changes[threadIdx.x] = local_changed;
            __syncthreads();

            if (threadIdx.x == 0) {
                block_changed_shared = false;
                for (int i = 0; i < block_size; i++) {
                    if (local_changes[i]) {
                        block_changed_shared = true;
                        break;
                    }
                }
                changed = block_changed_shared;
                iter++;
            }
            __syncthreads();
        }
    };

    auto do_label_cleanup = [&] __device__() {
        for (int i = start_idx; i < end_idx; i++)
            if (i < total) d_label_counts[i] = 0;
        __syncthreads();

        for (int i = start_idx; i < end_idx;) {
            if (i >= total) break;
            int lab = d_labels[i];
            if (lab == BACKGROUND) {
                i++;
                continue;
            }
            int run_length = 1;
            int j = i + 1;
            while (j < end_idx && j < total && d_labels[j] == lab) {
                run_length++;
                j++;
            }
            atomicAdd(&d_label_counts[lab], run_length);
            i = j;
        }
        __syncthreads();

        for (int i = start_idx; i < end_idx; i++) {
            if (i < total) {
                int lab = d_labels[i];
                if (lab != BACKGROUND && d_label_counts[lab] < cnt_threshold) d_labels[i] = BACKGROUND;
            }
        }
        __syncthreads();
    };

    do_lable_propagation();
    do_label_cleanup();
}

bool file_exists(const std::string &name) {
    std::fstream fs(name.c_str(), std::ios::in);
    return fs.good();
}

inline json load_json(std::string fn) {
    if (file_exists(fn)) {
        std::ifstream ifs(fn);
        return json::parse(ifs);
    }
    else {
        json j = {};
        return j;
    }
}

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {
    for (int p = 0; p < 4; ++p) {
        auto label = load_json("../label_propagation/label_" + std::to_string(p) + ".json");
        auto connect_flag = load_json("../label_propagation/connect_flag" + std::to_string(p) + ".json");
        auto pre_label = load_json("../label_propagation/pre_label_counts" + std::to_string(p) + ".json");

        thrust::host_vector<int> label_v = label.get<thrust::host_vector<int>>();
        thrust::host_vector<uint8_t> connect_flag_v = connect_flag.get<thrust::host_vector<uint8_t>>();
        thrust::host_vector<int> pre_label_v = pre_label.get<thrust::host_vector<int>>();

        thrust::device_vector<int> d_label = label_v;
        thrust::device_vector<uint8_t> d_connect_flag = connect_flag_v;
        thrust::device_vector<int> d_pre_label = pre_label_v;

        int h = 94;
        int w = 32;
        int max_iter = 10;
        int min_obj_size = 16;
        int cuda_block_size = 128;

        label_propagation_full<<<1, cuda_block_size>>>(d_label.data().get(), d_pre_label.data().get(), d_connect_flag.data().get(), h, w, max_iter, min_obj_size);
    }
}