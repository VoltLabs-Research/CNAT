#include <volt/plugin/cnat_model.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace Volt {

extern const CnatModel::Embedded kCnatEmbeddedModels[];
extern const std::size_t kCnatEmbeddedModelCount;

namespace {

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> bytes) : _bytes(bytes) {}

    std::uint32_t u32() {
        std::uint8_t b[4];
        take(b, 4);
        return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8)
            | (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
    }

    std::string text(std::size_t size) {
        std::string s(size, '\0');
        take(s.data(), size);
        return s;
    }

    std::vector<float> floats(std::size_t count) {
        std::vector<float> v(count);
        for (auto& f : v) {
            const std::uint32_t bits = u32();
            std::memcpy(&f, &bits, sizeof(f));
        }
        return v;
    }

    bool done() const { return _offset == _bytes.size(); }

private:
    void take(void* dst, std::size_t n) {
        if (_offset + n > _bytes.size())
            throw std::runtime_error("CNAT model file is truncated");
        std::memcpy(dst, _bytes.data() + _offset, n);
        _offset += n;
    }

    std::span<const std::uint8_t> _bytes;
    std::size_t _offset = 0;
};

inline void layerNorm(const float* x, const std::vector<float>& gain, const std::vector<float>& bias,
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

// The plugin ships one x86-64 binary built for the baseline ISA; on Linux the dense layers, about 85%
// of the work, also get an AVX2/FMA clone that the loader picks on CPUs that have it.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__linux__)
#define CNAT_MULTIVERSION __attribute__((target_clones("arch=haswell", "default")))
#else
#define CNAT_MULTIVERSION
#endif

// out[r][c0..c1) = bias + in[r] . W[:, c0..c1) for weightT stored [in][out]. Rows go in blocks of
// eight so each weight row is read once per block instead of once per token.
CNAT_MULTIVERSION
void linearRows(const float* weightT, const float* bias, int in, int outDim, const float* x, int ldIn,
                int rows, float* out, int ldOut, int c0, int c1) {
    constexpr int kRowBlock = 8;
    for (int r0 = 0; r0 < rows; r0 += kRowBlock) {
        const int nr = std::min(kRowBlock, rows - r0);
        for (int r = 0; r < nr; ++r) {
            float* y = out + static_cast<std::size_t>(r0 + r) * ldOut;
            for (int c = c0; c < c1; ++c) y[c] = bias[c];
        }
        for (int i = 0; i < in; ++i) {
            const float* w = weightT + static_cast<std::size_t>(i) * outDim;
            for (int r = 0; r < nr; ++r) {
                const float ai = x[static_cast<std::size_t>(r0 + r) * ldIn + i];
                float* y = out + static_cast<std::size_t>(r0 + r) * ldOut;
                for (int c = c0; c < c1; ++c) y[c] += ai * w[c];
            }
        }
    }
}

inline float gelu(float x) {
    // tanh approximation, as flax.linen.gelu and torch gelu(approximate="tanh")
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

}

CnatModel CnatModel::fromBytes(std::span<const std::uint8_t> bytes) {
    Reader r(bytes);
    if (r.text(4) != "CNAT")
        throw std::runtime_error("not a CNAT model file (bad magic)");
    if (const auto version = r.u32(); version != 1)
        throw std::runtime_error("unsupported CNAT model version " + std::to_string(version));

    CnatModel m;
    auto& c = m._config;
    c.hidden = static_cast<int>(r.u32());
    c.layers = static_cast<int>(r.u32());
    c.heads = static_cast<int>(r.u32());
    c.mlp = static_cast<int>(r.u32());
    c.classes = static_cast<int>(r.u32());
    c.neighbors = static_cast<int>(r.u32());
    if (c.hidden <= 0 || c.layers <= 0 || c.heads <= 0 || c.hidden % c.heads != 0 || c.mlp <= 0
        || c.classes < 2 || c.neighbors <= 0 || c.neighbors > 127)
        throw std::runtime_error("invalid CNAT model configuration");
    m._metadata = r.text(r.u32());

    auto linear = [&](int in, int out) {
        Linear l;
        l.in = in;
        l.out = out;
        const auto w = r.floats(static_cast<std::size_t>(in) * out);
        l.weightT.resize(w.size());
        for (int o = 0; o < out; ++o)
            for (int i = 0; i < in; ++i)
                l.weightT[static_cast<std::size_t>(i) * out + o] = w[static_cast<std::size_t>(o) * in + i];
        l.bias = r.floats(out);
        m._parameterCount += w.size() + l.bias.size();
        return l;
    };
    auto norm = [&](int d) {
        LayerNorm n{r.floats(d), r.floats(d)};
        m._parameterCount += 2 * static_cast<std::size_t>(d);
        return n;
    };

    const int d = c.hidden;
    m._embed = linear(3, d);
    m._cls = r.floats(d);
    m._parameterCount += d;
    for (int l = 0; l < c.layers; ++l) {
        Block b;
        b.ln1 = norm(d);
        b.qkv = linear(d, 3 * d);
        b.proj = linear(d, d);
        b.ln2 = norm(d);
        b.fc1 = linear(d, c.mlp);
        b.fc2 = linear(c.mlp, d);
        m._blocks.push_back(std::move(b));
    }
    m._norm = norm(d);
    m._head = linear(d, c.classes);
    if (!r.done())
        throw std::runtime_error("CNAT model file has trailing bytes");
    return m;
}

CnatModel CnatModel::fromFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open CNAT model " + path);
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    return fromBytes(bytes);
}

std::span<const CnatModel::Embedded> CnatModel::embeddedModels() {
    return {kCnatEmbeddedModels, kCnatEmbeddedModelCount};
}

CnatModel CnatModel::load(const std::string& nameOrPath) {
    for (const Embedded& e : embeddedModels())
        if (nameOrPath.empty() || nameOrPath == e.name)
            return fromBytes({e.data, e.size});
    return fromFile(nameOrPath);
}

void CnatModel::classify(const float* neighbors, Workspace& ws, float* probabilities) const {
    const int d = _config.hidden;
    const int n = _config.neighbors + 1;
    const int heads = _config.heads;
    const int hd = d / heads;
    const int d3 = 3 * d;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    ws.x.resize(static_cast<std::size_t>(n) * d);
    ws.h.resize(static_cast<std::size_t>(n) * d);
    ws.qkv.resize(static_cast<std::size_t>(n) * d3);
    ws.attention.resize(static_cast<std::size_t>(n) * d);
    ws.hidden.resize(static_cast<std::size_t>(n) * _config.mlp);
    ws.scores.resize(n);
    float* x = ws.x.data();

    auto apply = [](const Linear& l, const float* in, int ldIn, int rows, float* out, int ldOut, int c0, int c1) {
        linearRows(l.weightT.data(), l.bias.data(), l.in, l.out, in, ldIn, rows, out, ldOut, c0, c1);
    };

    float maxNorm = 0.0f;
    for (int t = 0; t < n - 1; ++t) {
        const float* v = neighbors + 3 * t;
        maxNorm = std::max(maxNorm, v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    const float inv = maxNorm > 0.0f ? 1.0f / std::sqrt(maxNorm) : 1.0f;
    float scaled[3 * 128];
    for (int i = 0; i < 3 * (n - 1); ++i) scaled[i] = neighbors[i] * inv;

    std::copy(_cls.begin(), _cls.end(), x);
    apply(_embed, scaled, 3, n - 1, x + d, d, 0, d);

    for (std::size_t l = 0; l < _blocks.size(); ++l) {
        const Block& b = _blocks[l];
        // Only the CLS row is read after the last block, so its queries and MLP run for row 0 only.
        const int rows = l + 1 == _blocks.size() ? 1 : n;

        for (int t = 0; t < n; ++t)
            layerNorm(x + static_cast<std::size_t>(t) * d, b.ln1.gain, b.ln1.bias, d, ws.h.data() + static_cast<std::size_t>(t) * d);
        apply(b.qkv, ws.h.data(), d, rows, ws.qkv.data(), d3, 0, d);
        apply(b.qkv, ws.h.data(), d, n, ws.qkv.data(), d3, d, d3);

        for (int i = 0; i < rows; ++i) {
            for (int hh = 0; hh < heads; ++hh) {
                const float* q = ws.qkv.data() + static_cast<std::size_t>(i) * d3 + hh * hd;
                float maxScore = -INFINITY;
                for (int j = 0; j < n; ++j) {
                    const float* k = ws.qkv.data() + static_cast<std::size_t>(j) * d3 + d + hh * hd;
                    float s = 0.0f;
                    for (int c = 0; c < hd; ++c) s += q[c] * k[c];
                    ws.scores[j] = s * scale;
                    maxScore = std::max(maxScore, ws.scores[j]);
                }
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    ws.scores[j] = std::exp(ws.scores[j] - maxScore);
                    sum += ws.scores[j];
                }
                float* o = ws.attention.data() + static_cast<std::size_t>(i) * d + hh * hd;
                std::fill(o, o + hd, 0.0f);
                for (int j = 0; j < n; ++j) {
                    const float p = ws.scores[j] / sum;
                    const float* v = ws.qkv.data() + static_cast<std::size_t>(j) * d3 + 2 * d + hh * hd;
                    for (int c = 0; c < hd; ++c) o[c] += p * v[c];
                }
            }
        }
        apply(b.proj, ws.attention.data(), d, rows, ws.h.data(), d, 0, d);
        for (int i = 0; i < rows * d; ++i) x[i] += ws.h[i];

        for (int t = 0; t < rows; ++t)
            layerNorm(x + static_cast<std::size_t>(t) * d, b.ln2.gain, b.ln2.bias, d, ws.h.data() + static_cast<std::size_t>(t) * d);
        apply(b.fc1, ws.h.data(), d, rows, ws.hidden.data(), _config.mlp, 0, _config.mlp);
        for (int i = 0; i < rows * _config.mlp; ++i) ws.hidden[i] = gelu(ws.hidden[i]);
        apply(b.fc2, ws.hidden.data(), _config.mlp, rows, ws.h.data(), d, 0, d);
        for (int i = 0; i < rows * d; ++i) x[i] += ws.h[i];
    }

    layerNorm(x, _norm.gain, _norm.bias, d, ws.h.data());
    apply(_head, ws.h.data(), d, 1, probabilities, _config.classes, 0, _config.classes);
    float maxLogit = *std::max_element(probabilities, probabilities + _config.classes);
    float sum = 0.0f;
    for (int c = 0; c < _config.classes; ++c) {
        probabilities[c] = std::exp(probabilities[c] - maxLogit);
        sum += probabilities[c];
    }
    for (int c = 0; c < _config.classes; ++c) probabilities[c] /= sum;
}

}
