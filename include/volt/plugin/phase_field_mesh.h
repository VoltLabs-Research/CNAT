#pragma once

#include <volt/core/simulation_cell.h>
#include <volt/math/point3.h>

#include <array>
#include <vector>

namespace Volt {

struct PhaseFieldMesh {
    std::vector<Point3> vertices;
    std::vector<std::array<int, 3>> facets;
    double area = 0.0;
};

// Grid point (i, j, k) of a dims[0] x dims[1] x dims[2] grid sits at reduced coordinates
// ((i + 0.5) / dims[0], ...) and field[i + dims[0] * (j + dims[1] * k)] is its value.
inline Point3 gridPointReduced(const std::array<int, 3>& dims, double i, double j, double k) {
    return Point3((i + 0.5) / dims[0], (j + 0.5) / dims[1], (k + 0.5) / dims[2]);
}

// Separable Gaussian of standard deviation sigma (A) along each cell vector, periodic where the
// cell is and clamped at open faces. The CNAT field is nearly binary, so its raw iso-surface is a
// staircase of grid cells; a sigma of about the grid spacing removes it without moving the surface.
std::vector<float> smoothField(const std::vector<float>& field, const std::array<int, 3>& dims,
                               const SimulationCell& cell, double sigma);

// Surface field == iso by marching tetrahedra (six Freudenthal tetrahedra per grid cube), wrapped
// across periodic cell faces. Facet normals point from field >= iso towards field < iso. Vertices
// on the far side of a periodic face are kept unwrapped, so the mesh has seams on the cell faces.
PhaseFieldMesh extractIsosurface(const std::vector<float>& field, const std::array<int, 3>& dims,
                                 const SimulationCell& cell, float iso);

}
