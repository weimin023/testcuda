#pragma once
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <stdexcept>
#include <cublas_v2.h>
#include <functional>

#define def function
#define memblock_size (32)
using namespace std;

template <typename T> using VEC = std::vector<T>;
template <typename T> T muxt(bool sel, T a, T b) { return sel ? a : b; }

class DatShape {
public:
    DatShape() : n_row(0), n_col(0), n_height(0), size(0), dim(0) {}
    DatShape(size_t n_row, size_t n_col = 1, size_t n_height = 1) : n_row(n_row), n_col(n_col), n_height(n_height) {
        dim = muxt<size_t>(n_height == 1, muxt<size_t>(n_col == 1, 1, 2), 3);
        size = n_row * n_col * n_height;
    }

    auto &operator=(const DatShape &that) {
        n_row = that.n_row;
        n_col = that.n_col;
        n_height = that.n_height;
        size = that.size;
        dim = that.dim;
        return *this;
    }
    bool operator==(const DatShape &that) const { return n_row == that.n_row && n_col == that.n_col && n_height == that.n_height && size == that.size && dim == that.dim; }
    bool operator!=(const DatShape &that) const { return !(*this == that); }

    size_t n_row, n_col, n_height, size, dim;

private:
};



template <typename D> class cuData_t {
public:
    cuData_t() : m_shape_{}, m_internal_mem_(nullptr) {}

    cuData_t(const std::vector<D> &other, cudaStream_t stream_in = 0) { 
        allocate(other.size());
        __mmwrite__(other.data(), size(), cudaMemcpyHostToDevice, 0, stream_in);
    }

    cuData_t(uint n_row) { allocate(n_row, n_col, n_height); }

    cuData_t(uint n_row, D val) {
        allocate(n_row);
        set_value(val);
    }

    cuData_t(const cuData_t &other, bool read_only = false) : m_read_only(read_only), m_using_cuda_malloc(other.m_using_cuda_malloc) {
        if (read_only) {
            m_internal_mem_ = other.data();
            m_shape_ = other.m_shape_;
        } else {
            assign_by_ref(other);
        }
    }    
    ~cuData_t() { free(); }

    uint size() const { return m_shape_.size; }

    key_t get_sn() const { return m_key; }

    static cuData_t create_Managed(uint n_row = 0) { return cuData_t(DatShape(n_row), false); }

    void set_key(key_t key) { this->m_key = key; }
    key_t get_key() const { return m_key; }

    cuData_t &operator=(const cuData_t &that) { return assign_by_ref(that); }

    void allocate(const DatShape &new_shape) {
        def malloc = [&]() {
            if (m_using_cuda_malloc) return cudaMalloc((void **)&m_internal_mem_, m_capacity * sizeof(D));
            return cudaMallocManaged((void **)&m_internal_mem_, m_capacity * sizeof(D));
        };

        if (m_read_only) throw std::runtime_error(_set_error_message("allocate: read_only"));

        if ((new_shape.size > m_capacity) && m_construct_by_handler) { throw std::runtime_error(_set_error_message("allocate by handler: out of range, new_shape.size > _capacity")); }
        if ((new_shape.size > m_capacity)) {
            free();
            m_capacity = ((new_shape.size + m_memblock_size_ - 1) / m_memblock_size_) * m_memblock_size_;
            if (m_capacity > 0) (malloc());
        }
        m_shape_ = new_shape;
    }

    void allocate_GPU(size_t n_row, size_t n_col, D *val) {
        cudaMalloc((void **)&m_internal_mem_, n_row * n_col * sizeof(D));
        cudaMemcpy(m_internal_mem_, val, n_row * n_col * sizeof(D), cudaMemcpyDeviceToDevice);
    }

    void allocate(size_t n_row, size_t n_col = 1, size_t n_height = 1) { allocate(DatShape(n_row, n_col, n_height)); }
    void resize(size_t n_row, size_t n_col = 1, size_t n_height = 1) { allocate(n_row, n_col, n_height); }

    cuData_t assign_by_vec(const std::vector<D> &other, cudaStream_t stream = 0) {
        allocate(other.size());
        __mmwrite__(other.data(), size(), cudaMemcpyHostToDevice, 0, stream);
        return *this;
    }

    cuData_t &assign_by_ref(const cuData_t &other, cudaStream_t stream = 0) {
        allocate(other.m_shape_);
        __mmwrite__(other.data(), size(), cudaMemcpyDeviceToDevice, 0, stream);
        assign_params(other);

        return *this;
    }

    void assign_by_host_ptr(const D *other, size_t len, cudaStream_t stream = 0) {
        allocate(len);
        __mmwrite__(other, len, cudaMemcpyHostToDevice, 0, stream);
    }

    cuData_t copy() {
        cuData_t out = (*this);
        return out;
    }
    void fill(D value, cudaStream_t stream = 0) { set_value(value, 0, size()); }

    template <typename H> VEC<H> to_host_vec(int offset = 0, int len = 0) const {
        if (len == 0) len = size() - offset;
        VEC<H> out(len);
        if (typeid(H) != typeid(D)) throw std::runtime_error(_set_error_message("to_host_vec: type mismatch"));
        cudaMemcpy(out.data(), m_internal_mem_ + offset, len * sizeof(D), cudaMemcpyDeviceToHost);
        return out;
    }
    std::vector<D> to_host() const {
        int len = size();
        std::vector<D> out(len);
        cudaMemcpy(out.data(), m_internal_mem_, len * sizeof(D), cudaMemcpyDeviceToHost);
        return out;
    };

    void free() {
        if (m_construct_by_handler) (cudaIpcCloseMemHandle(m_internal_mem_));
        if (!m_read_only && !m_construct_by_handler && m_internal_mem_ != nullptr && m_capacity) cudaFree(m_internal_mem_);
        m_shape_ = DatShape();
        m_capacity = 0;
        m_internal_mem_ = nullptr;
    }
    void clear() {
        if (!m_read_only && !m_construct_by_handler && m_internal_mem_ != nullptr && m_capacity) cudaFree(m_internal_mem_);
        m_shape_ = DatShape();
        m_capacity = 0;
        m_internal_mem_ = nullptr;
    }

    const size_t &n_row = m_shape_.n_row, &n_col = m_shape_.n_col, &n_height = m_shape_.n_height, &dim = m_shape_.dim;
    D *data() const { return m_internal_mem_; }

private:
    cudaStream_t stream = 0;

    void set_value(D value, size_t offset = 0, size_t target_size = 0) {
        if (target_size == 0) target_size = (size() - offset);
        cudaMemset(m_internal_mem_ + offset, value, target_size * sizeof(D));
    }

    void assign_params(const cuData_t &other) {
        if (!m_read_only) {
            m_shape_ = other.m_shape_;
            m_capacity = other.m_capacity;
            m_key = other.m_key;
            m_using_cuda_malloc = other.m_using_cuda_malloc;
        }
    }

    string _set_error_message(const string &errmsg) const {
        string type_name = typeid(D).name();
        string out = "cuData_t(" + type_name + ")::" + errmsg;
        return out;
    }

    void __mmwrite__(const D *data, int len, cudaMemcpyKind kind, int offset, cudaStream_t stream) {
        if (m_read_only) {
            throw std::runtime_error(_set_error_message("write: read_only"));
        }
        if (offset + len > m_capacity) { 
            throw std::runtime_error(_set_error_message("write : out of range")); 
        }
        m_shape_.size = max(m_shape_.size, (size_t)(offset + len));
        cudaMemcpyAsync(m_internal_mem_ + offset, data, len * sizeof(D), kind, (stream == 0) ? this->stream : stream);
    }

    void __mmread__(D *data, int len, cudaMemcpyKind kind, int offset, cudaStream_t stream) const { cudaMemcpyAsync(data, m_internal_mem_ + offset, len * sizeof(D), kind, (stream == 0) ? this->stream : stream); }
    void read_data(D *out, size_t len, size_t offset = 0, cudaStream_t stream = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) const {
        if (offset + len > size()) throw std::runtime_error(_set_error_message("read_data: out of range"));
        __mmread__(out, len, kind, offset, stream);
    }
    void read_data(VEC<D> &out) const {
        out.resize(size());
        read_data(out.data(), size(), 0, 0, cudaMemcpyDeviceToHost);
    }

    void read_data(cuData_t &out, cudaStream_t stream = 0) const {
        out.allocate(m_shape_);
        read_data(out.data(), out.size(), stream, cudaMemcpyDeviceToDevice);
    }

    bool m_construct_by_handler = false;
    bool m_using_cuda_malloc = true;
    bool m_read_only = false;

    D *m_internal_mem_ = nullptr;
    DatShape m_shape_{};
    
    const size_t m_memblock_size_ = memblock_size;
    size_t m_capacity = 0;

    key_t m_key = 0;
    
};

template <typename T> using cuVec = cuData_t<T>;
using cuDataF = cuData_t<float>;
using cuDataI = cuData_t<int>;
using cuDataU = cuData_t<uint>;

using cuDataD = cuData_t<double>;
using cuDataC = cuData_t<cuComplex>;
using cuDataZ = cuData_t<cuDoubleComplex>;
using cuDataB = cuData_t<bool>;

template <> void cuData_t<float>::set_value(float value, size_t offset, size_t target_size);
template <> void cuData_t<double>::set_value(double value, size_t offset, size_t target_size);
template <> void cuData_t<cuComplex>::set_value(cuComplex value, size_t offset, size_t target_size);
template <> void cuData_t<cuDoubleComplex>::set_value(cuDoubleComplex value, size_t offset, size_t target_size);

using cuVecF = cuVec<float>;
using cuvecI = cuVec<int>;
using cuVecU = cuVec<uint>;
using cuVecD = cuVec<double>;
using cuVecC = cuVec<cuComplex>;
using cuVecZ = cuVec<cuDoubleComplex>;


cublasStatus_t gemm_cuDoubleComplex(cuDataZ &out, cublasHandle_t &handle, const cuDataZ &in1, cublasOperation_t trans1, int row1, int col1, const cuDataZ &in2, cublasOperation_t trans2, int row2, int col2, float alpha, float beta);

double cuComplexDistance(const cuDoubleComplex& z1, const cuDoubleComplex& z2);

std::vector<std::vector<cuDoubleComplex>> matrixMultiply(
    const std::vector<std::vector<cuDoubleComplex>>& A,
    const std::vector<std::vector<cuDoubleComplex>>& B);



