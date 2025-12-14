#pragma once
#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <deque>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <immintrin.h> 
#include <chrono>
#include <cassert>

// ThreadPool
class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = 0) {
        if (numThreads == 0) numThreads = std::thread::hardware_concurrency();
        stop = false;
        for (size_t i = 0; i < numThreads; i++) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop_front();
                    }
                    task();
                }
                });
        }
    }

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
        using return_type = typename std::result_of<F()>::type;
        auto taskPtr = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> res = taskPtr->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace_back([taskPtr]() { (*taskPtr)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& w : workers) if (w.joinable()) w.join();
    }

private:
    std::vector<std::thread> workers;
    std::deque<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};


// CPU AVX2 detection
inline bool cpu_supports_avx2() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] >= 7) {
        int info[4];
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0; // AVX2 EBX[5]
    }
    return false;
}


// Tensor3D class
template<typename T>
class Tensor3D {
public:
    int dim0, dim1, dim2;
    std::vector<T> data;

    Tensor3D(int d0, int d1, int d2) : dim0(d0), dim1(d1), dim2(d2), data((size_t)d0* d1* d2) {}

    inline T& at(int z, int y, int x) { return data[(size_t)z * dim1 * dim2 + (size_t)y * dim2 + x]; }
    inline const T& at(int z, int y, int x) const { return data[(size_t)z * dim1 * dim2 + (size_t)y * dim2 + x]; }

    // Scalar matmul
    Tensor3D matmul_slices_scalar(const Tensor3D& B, ThreadPool* pool = nullptr) const {
        assert(dim0 == B.dim0 && dim2 == B.dim1);
        Tensor3D out(dim0, dim1, B.dim2);
        std::fill(out.data.begin(), out.data.end(), T(0));

        auto worker = [&](int zStart, int zEnd) {
            for (int z = zStart; z < zEnd; ++z)
                for (int i = 0; i < dim1; ++i)
                    for (int k = 0; k < dim2; ++k) {
                        T tmp = at(z, i, k);
                        for (int j = 0; j < B.dim2; ++j)
                            out.at(z, i, j) += tmp * B.at(z, k, j);
                    }
            };

        if (pool) {
            int numThreads = std::max(1u, std::thread::hardware_concurrency());
            int slicesPerThread = (dim0 + numThreads - 1) / numThreads;
            std::vector<std::future<void>> futures;
            futures.reserve(numThreads);
            for (int t = 0; t < numThreads; t++) {
                int start = t * slicesPerThread;
                int end = std::min(start + slicesPerThread, dim0);
                if (start >= end) break;
                futures.push_back(pool->enqueue([=, &out, this] { worker(start, end); }));
            }
            for (auto& f : futures) f.get();
        }
        else worker(0, dim0);

        return out;
    }

    // Naive AVX2 matmul
    Tensor3D matmul_slices_avx2(const Tensor3D& B, ThreadPool* pool = nullptr) const {
        assert(dim0 == B.dim0 && dim2 == B.dim1);
        Tensor3D out(dim0, dim1, B.dim2);
        std::fill(out.data.begin(), out.data.end(), T(0));

        auto worker = [&](int zStart, int zEnd) {
            for (int z = zStart; z < zEnd; ++z) {
                for (int i = 0; i < dim1; ++i) {
                    for (int k = 0; k < dim2; ++k) {
                        T aval = at(z, i, k);
                        const T* bp = &B.data[(size_t)z * B.dim1 * B.dim2 + (size_t)k * B.dim2];
                        T* op = &out.data[(size_t)z * out.dim1 * out.dim2 + (size_t)i * out.dim2];

                        if constexpr (std::is_same<T, double>::value) {
                            int j = 0;
                            __m256d vaval = _mm256_set1_pd(aval);
                            for (; j + 4 <= B.dim2; j += 4) {
                                __m256d vb = _mm256_loadu_pd(bp + j);
                                __m256d vo = _mm256_loadu_pd(op + j);
                                __m256d vr = _mm256_fmadd_pd(vaval, vb, vo);
                                _mm256_storeu_pd(op + j, vr);
                            }
                            for (; j < B.dim2; ++j) op[j] += aval * bp[j];
                        }
                        else {
                            for (int j = 0; j < B.dim2; ++j) op[j] += aval * bp[j];
                        }
                    }
                }
            }
            };

        if (pool) {
            int numThreads = std::max(1u, std::thread::hardware_concurrency());
            int slicesPerThread = (dim0 + numThreads - 1) / numThreads;
            std::vector<std::future<void>> futures;
            futures.reserve(numThreads);
            for (int t = 0; t < numThreads; t++) {
                int start = t * slicesPerThread;
                int end = std::min(start + slicesPerThread, dim0);
                if (start >= end) break;
                futures.push_back(pool->enqueue([=, &out, this] { worker(start, end); }));
            }
            for (auto& f : futures) f.get();
        }
        else worker(0, dim0);

        return out;
    }

    // Blocked AVX2
    Tensor3D matmul_slices_avx2_blocked(const Tensor3D& B, ThreadPool* pool = nullptr) const {
        assert(dim0 == B.dim0 && dim2 == B.dim1);
        Tensor3D out(dim0, dim1, B.dim2);
        std::fill(out.data.begin(), out.data.end(), T(0));

        constexpr int BLOCK_K = 32;
        constexpr int BLOCK_J = 32;

        auto worker = [&](int zStart, int zEnd) {
            for (int z = zStart; z < zEnd; ++z) {
                for (int i = 0; i < dim1; ++i) {
                    for (int kk = 0; kk < dim2; kk += BLOCK_K) {
                        int kMax = std::min(kk + BLOCK_K, dim2);
                        for (int jj = 0; jj < B.dim2; jj += BLOCK_J) {
                            int jMax = std::min(jj + BLOCK_J, B.dim2);
                            for (int k = kk; k < kMax; ++k) {
                                T aval = at(z, i, k);
                                const T* bp = &B.data[(size_t)z * B.dim1 * B.dim2 + (size_t)k * B.dim2 + jj];
                                T* op = &out.data[(size_t)z * out.dim1 * out.dim2 + (size_t)i * out.dim2 + jj];

                                if constexpr (std::is_same<T, double>::value) {
                                    int j = 0;
                                    __m256d vaval = _mm256_set1_pd(aval);
                                    for (; j + 4 <= jMax - jj; j += 4) {
                                        __m256d vb = _mm256_loadu_pd(bp + j);
                                        __m256d vo = _mm256_loadu_pd(op + j);
                                        __m256d vr = _mm256_fmadd_pd(vaval, vb, vo);
                                        _mm256_storeu_pd(op + j, vr);
                                    }
                                    for (; j < jMax - jj; ++j) op[j] += aval * bp[j];
                                }
                                else {
                                    for (int j = 0; j < jMax - jj; ++j) op[j] += aval * bp[j];
                                }
                            }
                        }
                    }
                }
            }
            };

        if (pool) {
            int numThreads = std::max(1u, std::thread::hardware_concurrency());
            int slicesPerThread = (dim0 + numThreads - 1) / numThreads;
            std::vector<std::future<void>> futures;
            futures.reserve(numThreads);
            for (int t = 0; t < numThreads; t++) {
                int start = t * slicesPerThread;
                int end = std::min(start + slicesPerThread, dim0);
                if (start >= end) break;
                futures.push_back(pool->enqueue([=, &out, this] { worker(start, end); }));
            }
            for (auto& f : futures) f.get();
        }
        else worker(0, dim0);

        return out;
    }

    // Blocked optimized AVX2
    Tensor3D matmul_slices_avx2_blocked_opt(const Tensor3D& B, ThreadPool* pool = nullptr) const {
        assert(dim0 == B.dim0 && dim2 == B.dim1);
        Tensor3D out(dim0, dim1, B.dim2);
        std::fill(out.data.begin(), out.data.end(), T(0));

        constexpr int BLOCK_K = 32;
        constexpr int BLOCK_J = 64;

        auto worker = [&](int zStart, int zEnd) {
            for (int z = zStart; z < zEnd; ++z) {
                for (int i = 0; i < dim1; ++i) {
                    for (int kk = 0; kk < dim2; kk += BLOCK_K) {
                        int kMax = std::min(kk + BLOCK_K, dim2);
                        for (int jj = 0; jj < B.dim2; jj += BLOCK_J) {
                            int jMax = std::min(jj + BLOCK_J, B.dim2);
                            for (int k = kk; k < kMax; ++k) {
                                T aval = at(z, i, k);
                                const T* bp = &B.data[(size_t)z * B.dim1 * B.dim2 + (size_t)k * B.dim2 + jj];
                                T* op = &out.data[(size_t)z * out.dim1 * out.dim2 + (size_t)i * out.dim2 + jj];

                                _mm_prefetch((const char*)(bp + 16), _MM_HINT_T0);

                                int j = 0;
                                __m256d vaval = _mm256_set1_pd(aval);
                                for (; j + 8 <= jMax - jj; j += 8) {
                                    __m256d vb0 = _mm256_loadu_pd(bp + j);
                                    __m256d vb1 = _mm256_loadu_pd(bp + j + 4);
                                    __m256d vo0 = _mm256_loadu_pd(op + j);
                                    __m256d vo1 = _mm256_loadu_pd(op + j + 4);

                                    __m256d vr0 = _mm256_fmadd_pd(vaval, vb0, vo0);
                                    __m256d vr1 = _mm256_fmadd_pd(vaval, vb1, vo1);

                                    _mm256_storeu_pd(op + j, vr0);
                                    _mm256_storeu_pd(op + j + 4, vr1);
                                }
                                for (; j < jMax - jj; ++j) op[j] += aval * bp[j];
                            }
                        }
                    }
                }
            }
            };

        if (pool) {
            int numThreads = std::max(1u, std::thread::hardware_concurrency());
            int slicesPerThread = (dim0 + numThreads - 1) / numThreads;
            std::vector<std::future<void>> futures;
            futures.reserve(numThreads);
            for (int t = 0; t < numThreads; t++) {
                int start = t * slicesPerThread;
                int end = std::min(start + slicesPerThread, dim0);
                if (start >= end) break;
                futures.push_back(pool->enqueue([=, &out, this] { worker(start, end); }));
            }
            for (auto& f : futures) f.get();
        }
        else worker(0, dim0);

        return out;
    }
};
