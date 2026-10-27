#include <volt/plugin/phase_field_mesh.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace Volt {

namespace {

// Corner c of a unit cube is offset (c & 1, (c >> 1) & 1, (c >> 2) & 1). Each tetrahedron walks from
// corner 0 to corner 7 adding one axis at a time; the six axis orders tile the cube, and the shared
// face diagonals match between neighbouring cubes.
constexpr int kTetrahedra[6][4] = {
    {0, 1, 3, 7}, {0, 1, 5, 7}, {0, 2, 3, 7}, {0, 2, 6, 7}, {0, 4, 5, 7}, {0, 4, 6, 7},
};

}

std::vector<float> smoothField(const std::vector<float>& field, const std::array<int, 3>& dims,
                               const SimulationCell& cell, double sigma) {
    std::vector<float> a = field, b(field.size());
    const std::array<std::size_t, 3> stride{1, static_cast<std::size_t>(dims[0]),
                                           static_cast<std::size_t>(dims[0]) * dims[1]};
    for (int d = 0; d < 3; ++d) {
        const double spacing = cell.matrix().column(d).length() / dims[d];
        const double s = sigma / spacing;
        const int radius = std::min(static_cast<int>(std::ceil(3.0 * s)), dims[d] / 2);
        if (s <= 0.0 || radius == 0)
            continue;
        std::vector<double> kernel(2 * radius + 1);
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t)
            sum += kernel[t + radius] = std::exp(-0.5 * t * t / (s * s));
        for (auto& w : kernel) w /= sum;

        const bool periodic = cell.pbcFlags()[d];
        for (std::size_t g = 0; g < a.size(); ++g) {
            const int i = static_cast<int>((g / stride[d]) % dims[d]);
            const std::size_t base = g - static_cast<std::size_t>(i) * stride[d];
            double acc = 0.0;
            for (int t = -radius; t <= radius; ++t) {
                int j = i + t;
                j = periodic ? ((j % dims[d]) + dims[d]) % dims[d] : std::clamp(j, 0, dims[d] - 1);
                acc += kernel[t + radius] * a[base + static_cast<std::size_t>(j) * stride[d]];
            }
            b[g] = static_cast<float>(acc);
        }
        std::swap(a, b);
    }
    return a;
}

PhaseFieldMesh extractIsosurface(const std::vector<float>& field, const std::array<int, 3>& dims,
                                 const SimulationCell& cell, float iso) {
    PhaseFieldMesh mesh;
    const auto& pbc = cell.pbcFlags();
    std::array<int, 3> cubes{};
    for (int d = 0; d < 3; ++d)
        cubes[d] = pbc[d] ? dims[d] : dims[d] - 1;
    if (cubes[0] <= 0 || cubes[1] <= 0 || cubes[2] <= 0)
        return mesh;

    // Unwrapped grid indices run 0..dims[d]; index dims[d] is the periodic image of index 0.
    const std::uint64_t sx = static_cast<std::uint64_t>(dims[0]) + 1;
    const std::uint64_t sy = static_cast<std::uint64_t>(dims[1]) + 1;
    const std::uint64_t pointCount = sx * sy * (static_cast<std::uint64_t>(dims[2]) + 1);
    std::unordered_map<std::uint64_t, int> edgeVertex;

    auto value = [&](int i, int j, int k) {
        i %= dims[0];
        j %= dims[1];
        k %= dims[2];
        return field[static_cast<std::size_t>(i) + static_cast<std::size_t>(dims[0]) * (j + static_cast<std::size_t>(dims[1]) * k)];
    };

    for (int k = 0; k < cubes[2]; ++k) {
        for (int j = 0; j < cubes[1]; ++j) {
            for (int i = 0; i < cubes[0]; ++i) {
                std::array<int, 3> corner[8];
                float f[8];
                std::uint64_t id[8];
                int insideCount = 0;
                for (int c = 0; c < 8; ++c) {
                    corner[c] = {i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)};
                    f[c] = value(corner[c][0], corner[c][1], corner[c][2]);
                    id[c] = corner[c][0] + sx * (corner[c][1] + sy * static_cast<std::uint64_t>(corner[c][2]));
                    insideCount += f[c] >= iso;
                }
                if (insideCount == 0 || insideCount == 8)
                    continue;

                auto position = [&](int c) {
                    return cell.reducedToAbsolute(gridPointReduced(dims, corner[c][0], corner[c][1], corner[c][2]));
                };
                auto crossing = [&](int a, int b) {
                    const std::uint64_t lo = std::min(id[a], id[b]);
                    const std::uint64_t hi = std::max(id[a], id[b]);
                    const auto [it, inserted] = edgeVertex.try_emplace(lo * pointCount + hi, static_cast<int>(mesh.vertices.size()));
                    if (inserted) {
                        const double t = (iso - f[a]) / static_cast<double>(f[b] - f[a]);
                        const Point3 pa = position(a);
                        mesh.vertices.push_back(pa + (position(b) - pa) * t);
                    }
                    return it->second;
                };

                for (const auto& tet : kTetrahedra) {
                    int in[4], out[4], nIn = 0, nOut = 0;
                    for (int c : tet)
                        (f[c] >= iso ? in[nIn++] : out[nOut++]) = c;
                    if (nIn == 0 || nOut == 0)
                        continue;

                    const Point3 p0 = position(tet[0]);
                    Vector3 towardsOutside = Vector3::Zero();
                    for (int a = 0; a < nOut; ++a) towardsOutside += (position(out[a]) - p0) / nOut;
                    for (int a = 0; a < nIn; ++a) towardsOutside -= (position(in[a]) - p0) / nIn;

                    auto emit = [&](int a, int b, int c) {
                        const Vector3 n = (mesh.vertices[b] - mesh.vertices[a]).cross(mesh.vertices[c] - mesh.vertices[a]);
                        if (n.dot(towardsOutside) < 0)
                            std::swap(b, c);
                        mesh.facets.push_back({a, b, c});
                        mesh.area += 0.5 * n.length();
                    };

                    if (nIn == 1 || nOut == 1) {
                        const int lone = nIn == 1 ? in[0] : out[0];
                        const int* rest = nIn == 1 ? out : in;
                        emit(crossing(lone, rest[0]), crossing(lone, rest[1]), crossing(lone, rest[2]));
                    } else {
                        const int ac = crossing(in[0], out[0]);
                        const int ad = crossing(in[0], out[1]);
                        const int bd = crossing(in[1], out[1]);
                        const int bc = crossing(in[1], out[0]);
                        emit(ac, ad, bd);
                        emit(ac, bd, bc);
                    }
                }
            }
        }
    }
    return mesh;
}

}
