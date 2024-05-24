
namespace cuMath {
template <typename T>
class cuVec {
   public:
    cuVec(size_t n_size, const T &n_val = 0) : n_size_(n_size) {
        cudaMalloc((void **)&deviceMem, n_size * sizeof(T));
        hostMem.resize(n_size, n_val);
        toDevice();
    }

    T &operator[](int i) {
        if (i < 0 || i >= n_size_) {
            throw std::out_of_range("Index out of bound");
        }
        return hostMem[i];
    }
    T &operator[](int i) const {
        if (i < 0 || i >= n_size_) {
            throw std::out_of_range("Index out of bound");
        }
        return hostMem[i];
    }

   private:
    void toDevice() {
        cudaMemcpy(deviceMem, hostMem.data(), n_size_ * sizeof(T),
                              cudaMemcpyHostToDevice);
    }

    void toHost() {
        cudaMemcpy(hostMem.data(), deviceMem, n_size_ * sizeof(T),
                              cudaMemcpyDeviceToHost);
    }

    T *deviceMem;
    std::vector<T> hostMem;
    size_t n_size_;
};
}  // namespace cuMath

