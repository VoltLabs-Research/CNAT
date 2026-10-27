// Checks extractIsosurface on fields with known surfaces:
//   sphere of radius 12 in a 40 A cube: area 4 pi R^2, closed, consistently oriented, volume 4/3 pi R^3
//   slab between reduced y = 0.25 and 0.75 in a periodic (sheared) cell: area 2 |a x c|
// With arguments, also reports the area and solid fraction of a raw float32 field at several smoothings:
//   cnat-mesh-check <field.f32> nx ny nz lx ly lz
#include <volt/plugin/phase_field_mesh.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numbers>
#include <utility>

using namespace Volt;

namespace {

SimulationCell makeCell(const Vector3& a, const Vector3& b, const Vector3& c) {
    SimulationCell cell;
    AffineTransformation m = AffineTransformation::Identity();
    for (int i = 0; i < 3; ++i) {
        m(i, 0) = a[i];
        m(i, 1) = b[i];
        m(i, 2) = c[i];
        m(i, 3) = 0.0;
    }
    cell.setMatrix(m);
    cell.setPbcFlags(true, true, true);
    return cell;
}

template<typename F>
std::vector<float> sample(const std::array<int, 3>& dims, const SimulationCell& cell, F f) {
    std::vector<float> field(static_cast<std::size_t>(dims[0]) * dims[1] * dims[2]);
    for (int k = 0; k < dims[2]; ++k)
        for (int j = 0; j < dims[1]; ++j)
            for (int i = 0; i < dims[0]; ++i) {
                const Point3 r = gridPointReduced(dims, i, j, k);
                field[i + dims[0] * (j + static_cast<std::size_t>(dims[1]) * k)] = f(r, cell.reducedToAbsolute(r));
            }
    return field;
}

bool check(const char* what, double got, double want, double tol) {
    const bool ok = std::abs(got - want) <= tol * std::abs(want);
    std::printf("%-44s %12.2f  expected %12.2f  %s\n", what, got, want, ok ? "ok" : "FAIL");
    return ok;
}

}

int main(int argc, char** argv) {
    if (argc == 8) {
        const std::array<int, 3> dims{std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])};
        const Vector3 a(std::atof(argv[5]), 0, 0), b(0, std::atof(argv[6]), 0), c(0, 0, std::atof(argv[7]));
        const SimulationCell cell = makeCell(a, b, c);
        std::vector<float> field(static_cast<std::size_t>(dims[0]) * dims[1] * dims[2]);
        std::FILE* in = std::fopen(argv[1], "rb");
        if (!in || std::fread(field.data(), sizeof(float), field.size(), in) != field.size())
            return 2;
        std::fclose(in);
        for (const double sigma : {0.0, 1.0, 2.0, 3.0}) {
            const auto f = sigma > 0 ? smoothField(field, dims, cell, sigma) : field;
            std::size_t solid = 0;
            for (float v : f) solid += v >= 0.5f;
            const PhaseFieldMesh mesh = extractIsosurface(f, dims, cell, 0.5f);
            std::printf("sigma %.0f A: area %9.1f A^2, area / 2|a x c| %.3f, solid fraction %.4f\n", sigma, mesh.area,
                        mesh.area / (2 * a.cross(c).length()), static_cast<double>(solid) / f.size());
        }
        return 0;
    }
    bool ok = true;
    {
        const SimulationCell cell = makeCell({40, 0, 0}, {0, 40, 0}, {0, 0, 40});
        const std::array<int, 3> dims{40, 40, 40};
        const double R = 12.0;
        const auto field = sample(dims, cell, [&](const Point3&, const Point3& p) {
            const double r = (p - Point3(20, 20, 20)).length();
            return static_cast<float>(0.5 * (1.0 + std::tanh((R - r) / 1.5)));
        });
        const PhaseFieldMesh mesh = extractIsosurface(field, dims, cell, 0.5f);

        std::map<std::pair<int, int>, int> directed;
        double volume = 0.0;
        for (const auto& f : mesh.facets) {
            for (int e = 0; e < 3; ++e) directed[{f[e], f[(e + 1) % 3]}]++;
            const Vector3 a = mesh.vertices[f[0]] - Point3(20, 20, 20);
            const Vector3 b = mesh.vertices[f[1]] - Point3(20, 20, 20);
            const Vector3 c = mesh.vertices[f[2]] - Point3(20, 20, 20);
            volume += a.dot(b.cross(c)) / 6.0;
        }
        int bad = 0;
        for (const auto& [edge, count] : directed) {
            const auto rev = directed.find({edge.second, edge.first});
            bad += count != 1 || rev == directed.end() || rev->second != 1;
        }
        ok &= check("sphere area (A^2)", mesh.area, 4 * std::numbers::pi * R * R, 0.01);
        ok &= check("sphere volume, outward normals (A^3)", volume, 4.0 / 3.0 * std::numbers::pi * R * R * R, 0.01);
        std::printf("%-44s %12d  expected %12d  %s\n", "sphere edges not shared by 2 opposite facets", bad, 0, bad ? "FAIL" : "ok");
        ok &= bad == 0;
    }
    for (const double shear : {0.0, 15.0}) {
        const Vector3 a(30, 0, 0), b(shear, 50, 0), c(0, 0, 20);
        const SimulationCell cell = makeCell(a, b, c);
        const std::array<int, 3> dims{15, 25, 10};
        const auto field = sample(dims, cell, [](const Point3& r, const Point3&) {
            return static_cast<float>(0.5 * (1.0 + std::tanh((0.25 - std::abs(r.y() - 0.5)) / 0.04)));
        });
        const PhaseFieldMesh mesh = extractIsosurface(field, dims, cell, 0.5f);
        ok &= check(shear == 0.0 ? "periodic slab area (A^2)" : "periodic slab area, sheared cell (A^2)",
                    mesh.area, 2 * a.cross(c).length(), 0.005);
    }
    return ok ? 0 : 1;
}
