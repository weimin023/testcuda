#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main() {
    const int N = 6;
    int h_keys[N] = {1, 4, 2, 8, 5, 7};
    char h_values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};

    // Allocate device memory
    int *d_keys;
    char *d_vals;
    cudaMalloc(&d_keys, N * sizeof(int));
    cudaMalloc(&d_vals, N * sizeof(char));

    // Copy data from host to device
    cudaMemcpy(d_keys, h_keys, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_values, N * sizeof(char), cudaMemcpyHostToDevice);

    // Perform the sort
    thrust::device_ptr<int> d_keys_ptr = thrust::device_pointer_cast(d_keys);
    thrust::device_ptr<char> d_vals_ptr = thrust::device_pointer_cast(d_vals);
    thrust::sort_by_key(d_keys_ptr, d_keys_ptr + N, d_vals_ptr);

    // Copy results back to the host
    cudaMemcpy(h_keys, d_keys, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, d_vals, N * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the sorted keys and values
    for (int i = 0; i < N; i++) {
        printf("Key: %d, Value: %c\n", h_keys[i], h_values[i]);
    }

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_vals);

    return 0;
}