#pragma once

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
