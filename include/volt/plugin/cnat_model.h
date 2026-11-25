#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Volt {

// CNAT (Watanabe, Sato & Shibuta, Comput. Mater. Sci. 276 (2027) 115172): the k nearest atoms of a
// point, divided by the largest distance, are embedded linearly and read out through a CLS token of a
// pre-norm Transformer encoder. There is no positional embedding, so neighbour order does not matter.
struct CnatConfig {
    int hidden = 0;
    int layers = 0;
    int heads = 0;
    int mlp = 0;
    int classes = 0;
    int neighbors = 0;
};

class CnatModel {
public:
    // Scratch buffers for one thread; classify() resizes them on first use.
    struct Workspace {
        std::vector<float> x, h, qkv, attention, hidden, scores;
    };

    // .cnat layout (little endian): "CNAT", u32 version = 1, u32 hidden, layers, heads, mlp, classes,
    // neighbors, u32 metadata size, metadata JSON, then f32 tensors in PyTorch [out][in] order:
    // embed.{w,b}, cls, per block ln1.{g,b} qkv.{w,b} proj.{w,b} ln2.{g,b} fc1.{w,b} fc2.{w,b}, norm.{g,b}, head.{w,b}.
    static CnatModel fromBytes(std::span<const std::uint8_t> bytes);
    static CnatModel fromFile(const std::string& path);

    // Models compiled into the binary (cmake/EmbedModel.cmake), the default first.
    struct Embedded {
        const char* name;
        const unsigned char* data;
        std::size_t size;
    };
    static std::span<const Embedded> embeddedModels();
    // An embedded model by name, the default for an empty name, else the .cnat file at that path.
    static CnatModel load(const std::string& nameOrPath);

    const CnatConfig& config() const { return _config; }
    const std::string& metadata() const { return _metadata; }
    std::size_t parameterCount() const { return _parameterCount; }

    // neighbors holds config().neighbors vectors (atom minus point, xyz); writes config().classes probabilities.
    void classify(const float* neighbors, Workspace& ws, float* probabilities) const;

private:
    // Reads a .cnat file's fields in order.
    class Reader;

    // The layers run per atom and per grid point, so the small ones are inline.
    static void layerNorm(const float* x, const std::vector<float>& gain, const std::vector<float>& bias,
                          int d, float* y) {
        float mean = 0.0f;
        for (int c = 0; c < d; ++c) mean += x[c];
        mean /= static_cast<float>(d);
        float var = 0.0f;
        for (int c = 0; c < d; ++c) {
            const float t = x[c] - mean;
            var += t * t;
        }
        const float inv = 1.0f / std::sqrt(var / static_cast<float>(d) + 1e-6f);
        for (int c = 0; c < d; ++c) y[c] = (x[c] - mean) * inv * gain[c] + bias[c];
    }

    // out[r][c0..c1) = bias + in[r] . W[:, c0..c1) for weightT stored [in][out]. Rows go in blocks of
    // eight so each weight row is read once per block instead of once per token.
    static void linearRows(const float* weightT, const float* bias, int in, int outDim, const float* x, int ldIn,
                           int rows, float* out, int ldOut, int c0, int c1);

    static float gelu(float x) {
        // tanh approximation, as flax.linen.gelu and torch gelu(approximate="tanh")
        return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }

    struct Linear {
        int in = 0;
        int out = 0;
        std::vector<float> weightT;  // [in][out]
        std::vector<float> bias;
    };
    struct LayerNorm {
        std::vector<float> gain;
        std::vector<float> bias;
    };
    struct Block {
        LayerNorm ln1;
        Linear qkv;
        Linear proj;
        LayerNorm ln2;
        Linear fc1;
        Linear fc2;
    };

    CnatConfig _config;
    std::string _metadata;
    std::size_t _parameterCount = 0;
    Linear _embed;
    std::vector<float> _cls;
    std::vector<Block> _blocks;
    LayerNorm _norm;
    Linear _head;
};

}
