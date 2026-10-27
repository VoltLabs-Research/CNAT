// Compares 1 - P(class 0) from CnatModel::classify with PyTorch's, from a fixture written by
// training/export_model.py --fixture.
//   cnat-parity-check <model.cnat> <fixture.bin>
#include <volt/plugin/cnat_model.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.cnat> <fixture.bin>\n", argv[0]);
        return 2;
    }
    const Volt::CnatModel model = Volt::CnatModel::fromFile(argv[1]);
    std::ifstream in(argv[2], std::ios::binary);
    std::uint32_t header[2];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    const std::uint32_t count = header[0], k = header[1];
    if (!in || static_cast<int>(k) != model.config().neighbors) {
        std::fprintf(stderr, "fixture does not match the model (k = %u)\n", k);
        return 2;
    }
    std::vector<float> atoms(static_cast<std::size_t>(count) * k * 3), expected(count);
    in.read(reinterpret_cast<char*>(atoms.data()), atoms.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(float));
    if (!in) {
        std::fprintf(stderr, "fixture is truncated\n");
        return 2;
    }

    Volt::CnatModel::Workspace ws;
    std::vector<float> p(model.config().classes);
    double maxDiff = 0.0;
    int labelMismatches = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < count; ++i) {
        model.classify(atoms.data() + static_cast<std::size_t>(i) * k * 3, ws, p.data());
        const double crystalline = 1.0 - static_cast<double>(p[0]);
        maxDiff = std::max(maxDiff, std::abs(crystalline - expected[i]));
        labelMismatches += (crystalline >= 0.5) != (expected[i] >= 0.5f);
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / count;
    std::printf("%u sets, max |dP| = %.3g, label mismatches %d, %.1f us per set (1 thread)\n",
                count, maxDiff, labelMismatches, us);
    return maxDiff < 1e-4 && labelMismatches == 0 ? 0 : 1;
}
