
﻿#include "Tensor3d.h"


// Benchmark 
template<typename Func>
double benchmark(Func f, int runs = 5) {
    double total = 0.0;
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return total / runs;
}


int main() {
    std::cout << "Tensor3D AVX2 benchmark\n";

    const int Z = 96, Y = 96, X = 96;
    Tensor3D<double> A(Z, Y, X);
    Tensor3D<double> B(Z, X, Y);

    for (int z = 0; z < Z; ++z)
        for (int y = 0; y < Y; ++y)
            for (int x = 0; x < X; ++x) {
                A.at(z, y, x) = z + y + x + 1;
                B.at(z, x, y) = (z + y + x + 1) * 0.5;
            }

    ThreadPool pool;

    double scalarTime = benchmark([&] {
        Tensor3D<double> C = A.matmul_slices_scalar(B, &pool);
        (void)C.at(0, 0, 0);
        });
    double avx2Time = benchmark([&] {
        Tensor3D<double> C = A.matmul_slices_avx2(B, &pool);
        (void)C.at(0, 0, 0);
        });
    double blockedTime = benchmark([&] {
        Tensor3D<double> C = A.matmul_slices_avx2_blocked(B, &pool);
        (void)C.at(0, 0, 0);
        });
    double blockedOptTime = benchmark([&] {
        Tensor3D<double> C = A.matmul_slices_avx2_blocked_opt(B, &pool);
        (void)C.at(0, 0, 0);
        });

   
    struct Result { std::string name; double time_ms; };
    std::vector<Result> results = {
        {"Scalar", scalarTime},
        {"Naive AVX2", avx2Time},
        {"Blocked AVX2", blockedTime},
        {"Blocked Optimized AVX2", blockedOptTime}
    };

    std::cout << "\n=== Benchmark Summary ===\n";
    for (auto& r : results) {
        double speedup = scalarTime / r.time_ms;
        double perc = speedup * 100.0;
        std::cout << r.name << ": "
            << r.time_ms << " ms, "
            << "Speedup: " << speedup << "× (" << perc << "%)\n";
    }


    std::cout << "\n=== Speedup Bar Chart (relative to Scalar) ===\n";
    const int maxBarWidth = 50;
    double maxSpeedup = scalarTime / blockedOptTime;

    for (auto& r : results) {
        double speedup = scalarTime / r.time_ms;
        int barLen = static_cast<int>(speedup / maxSpeedup * maxBarWidth);
        std::cout << r.name;
        if (r.name.length() < 20) std::cout << std::string(20 - r.name.length(), ' ');
        std::cout << " | ";
        for (int i = 0; i < barLen; i++) std::cout << "#";
        std::cout << " " << speedup << "×\n";
    }

    return 0;
}
