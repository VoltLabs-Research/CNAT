#pragma once

#include <volt/analysis/crystal_topology_library.h>
#include <volt/analysis/structure_analysis.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace Volt {

// Structure types of topologies that StructureType does not name; PTM numbers its user templates the
// same way.
inline constexpr int kCustomTopologyTypeBase = 1000;

// StructureType of a structure name such as "fcc" or "HCP" (case-insensitive); OTHER when StructureType
// has no such structure.
int structureTypeOfName(const std::string& name);

// A topology of the lattice registry (the YAML files OpenDXA resolves --reference_topology in): its
// lattice vectors, of which the first coordinationNumber form the canonical neighbour shell, the
// common neighbours of each shell vector and the proper rotations of the lattice; for matching, the
// shell scaled to unit mean length and the pair of its vectors every alignment starts from.
struct LatticeTopology {
    std::string name;
    int structureType = 0;
    int coordinationNumber = 0;
    std::vector<Vector3> latticeVectors;
    std::array<std::array<int, 2>, MAX_NEIGHBORS> commonNeighbors{};
    std::vector<SharedCrystalSymmetryPermutation> symmetries;
    // The bonds whose two common neighbours lie a lattice vector apart from them do not span space
    // (sc, st, a7), so clusters cannot grow from the shell overlap and join atoms by orientation instead.
    bool orientationClusters = false;
    // The rotations that map the shell onto itself.
    std::vector<int> shellSymmetries;
    std::vector<Vector3> unitVectors;
    int anchorA = 0;
    int anchorB = 1;
    Matrix3 anchorFrameTransposed = Matrix3(Matrix3::Identity{});
    // Site of a multi-site reference cell (loadCellTopologies), else -1. The sites of one cell share a
    // name and the crystal frame; atoms of different sites of one grain grow into one cluster.
    int cellSite = -1;
    // Atom type of the cell site, and of each of its neighbours: only atoms of the type take the site,
    // and each vector takes a neighbour of its type (cells of several species).
    int siteType = -1;
    std::vector<int> neighborTypes;
    // Cell site of each neighbour: a grain passes its sites on bond by bond (structure_contract.cpp).
    std::vector<int> neighborSites;
};

struct LatticeMatch {
    // In units of the mean neighbour distance, after the isotropic scale fit.
    double rmsd = 0.0;
    // Slot k holds the neighbour (index into the deltas) that sits on ideal vector k.
    std::array<int, MAX_NEIGHBORS> slots{};
    // Carries the ideal shell onto the neighbours in slot order.
    Matrix3 rotation = Matrix3(Matrix3::Identity{});
};

// Resolves the names in the lattice registry. Names StructureType has (fcc, hcp, bcc, sc,
// cubic_diamond, hex_diamond) keep its structure type, so their atoms read as the other producers'.
std::vector<LatticeTopology> loadLatticeTopologies(const std::vector<std::string>& names);

// Multi-site reference crystals, whose sites differ in environment (kyanite's Al: four orbits and their
// inversion images), cannot be one registry topology. Every site of the cell of these atom types becomes
// a topology of its own, named after the cell, whose shell is its neighbours of those types within the
// cutoff, in the frame of the cell. The cutoff is the given one or, for 0, the midpoint of the widest
// gap between neighbour distances of the cell that leaves every site 12 to MAX_NEIGHBORS neighbours.
// Only the identity is a symmetry of a site: the setting is fixed by which site an atom matches, and
// sites that a rotation or translation of the space group or a pseudo-symmetry relates fit alike, so the
// contract grows each grain from its best-aligned atom, passing its orientation and sites on bond by
// bond (structure_contract.cpp).
struct CellTopologies {
    std::string name;
    double cutoff = 0.0;
    std::vector<LatticeTopology> sites;
};
// A reference cell: edge vectors (columns), origin and atoms, from a LAMMPS data file (orthogonal or
// triclinic, 'atomic' or 'charge' style) or from a model, which stores the crystal of each of its
// crystal classes.
struct CellAtoms {
    Matrix3 cell = Matrix3(Matrix3::Zero{});
    Point3 origin = Point3::Origin();
    std::vector<int> types;
    std::vector<Point3> positions;
};
CellAtoms readLammpsData(const std::string& path);
// The sites of the atoms of these types (every atom for none) as topologies named name.
CellTopologies cellTopologies(const CellAtoms& data, std::string name, const std::vector<int>& types, double cutoff,
                              int firstStructureType);
// The same for a LAMMPS data file, named after its stem.
CellTopologies loadCellTopologies(const std::string& path, const std::vector<int>& types, double cutoff,
                                  int firstStructureType);

// Aligns the first coordinationNumber deltas, nearest first, onto the topology: of the proper
// rotations that carry the anchor pair onto a pair of neighbours, the one whose one-to-one assignment
// leaves the smallest residual, then refined by a least-squares rotation and scale. False when no
// rotation gives a one-to-one assignment.
// For a topology with neighbour types, deltaTypes gives the type of each of deltaCount deltas, nearest
// first and at least coordinationNumber of them: each vector takes the nearest of its type, so a shell
// that thermal motion reorders against the next one still aligns.
inline constexpr int kMaxMatchCandidates = 32;
bool matchLattice(const LatticeTopology& topology, const Vector3* deltas, LatticeMatch& match,
                  const int* deltaTypes = nullptr, int deltaCount = 0);
// The same from a given orientation instead of an anchor search: the assignment under the guess, then
// the refinement. Grows a grain from atoms whose orientation is known.
bool matchLatticeFrom(const LatticeTopology& topology, const Vector3* deltas, const Matrix3& guess, LatticeMatch& match,
                      const int* deltaTypes = nullptr, int deltaCount = 0);

// The steps matchLattice and matchLatticeFrom take, and the matching data of a topology. They run per site and per
// topology, so the small ones are inline.
namespace LatticeMatcherDetail {

// A neighbour pair is tried as the image of the anchor pair when its lengths and angle cosine are this
// close to the anchors'; the bounds only prune, the assignment residual decides.
inline constexpr double kAnchorLengthTolerance = 0.3;
inline constexpr double kAnchorCosineTolerance = 0.35;
inline constexpr int kRefinementPasses = 3;
inline constexpr int kPolarIterations = 30;

// The name in lower case: topologies and CNAT classes match by name whatever its case.
std::string lowercase(std::string s);

// Orthonormal frame (columns) with its first axis along a and its second in the a-b plane.
inline bool anchorFrame(const Vector3& a, const Vector3& b, Matrix3& frame) {
    const Vector3 normal = a.cross(b);
    const double normalLength = normal.length();
    if (!(normalLength > 1e-6 * a.length() * b.length())) return false;
    const Vector3 e1 = a / a.length();
    const Vector3 e3 = normal / normalLength;
    frame = Matrix3(e1, e3.cross(e1), e3);
    return true;
}

// Each scaled, rotated ideal vector takes its nearest neighbour of the candidates (of its type, with
// types); false when a neighbour is taken twice or a vector finds none.
inline bool assign(const LatticeTopology& topology, const Matrix3& rotation, double scale, const Vector3* v, int candidates,
                   const int* types, std::array<int, MAX_NEIGHBORS>& slots, double& residual) {
    const int cn = topology.coordinationNumber;
    std::uint32_t taken = 0;
    residual = 0.0;
    for (int k = 0; k < cn; ++k) {
        const Vector3 ideal = scale * (rotation * topology.unitVectors[k]);
        int nearest = -1;
        double nearestDistance = std::numeric_limits<double>::max();
        for (int j = 0; j < candidates; ++j) {
            if (types && types[j] != topology.neighborTypes[k]) continue;
            const double distance = (v[j] - ideal).squaredLength();
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearest = j;
            }
        }
        if (nearest < 0 || (taken & (1u << nearest))) return false;
        taken |= 1u << nearest;
        slots[k] = nearest;
        residual += nearestDistance;
    }
    return true;
}

// Rotation factor of the polar decomposition by Newton's iteration X <- (X + X^-T) / 2, where the
// columns of X^-T are the cross products of X's columns over det X. False unless det m > 0, i.e. unless
// the best orthogonal fit is a proper rotation.
inline bool rotationFactor(Matrix3 m, Matrix3& rotation) {
    if (!(m.determinant() > 1e-12)) return false;
    for (int i = 0; i < kPolarIterations; ++i) {
        const double det = m.determinant();
        const Matrix3 inverseTransposed(m.column(1).cross(m.column(2)) / det,
                                        m.column(2).cross(m.column(0)) / det,
                                        m.column(0).cross(m.column(1)) / det);
        const Matrix3 next = 0.5 * (m + inverseTransposed);
        double change = 0.0;
        for (int c = 0; c < 3; ++c) change += (next.column(c) - m.column(c)).squaredLength();
        m = next;
        if (change < 1e-24) break;
    }
    rotation = m;
    return true;
}

// The deltas in units of their mean length over the shell, and how many of them are candidates: the
// shell for a topology without neighbour types, else every delta given (matchLattice).
inline bool normalizeDeltas(const LatticeTopology& topology, const Vector3* deltas, const int* deltaTypes, int deltaCount,
                            std::array<Vector3, kMaxMatchCandidates>& v, int& candidates, const int*& types) {
    const int cn = topology.coordinationNumber;
    types = deltaTypes && !topology.neighborTypes.empty() ? deltaTypes : nullptr;
    candidates = types ? std::min(deltaCount, kMaxMatchCandidates) : cn;
    if (candidates < cn) return false;
    double meanLength = 0.0;
    for (int j = 0; j < cn; ++j) meanLength += deltas[j].length();
    meanLength /= cn;
    if (!(meanLength > 0.0)) return false;
    for (int j = 0; j < candidates; ++j) v[j] = deltas[j] / meanLength;
    return true;
}

// Least-squares rotation and scale for the assignment, then the assignment again under them; each
// step can only lower the residual.
inline void refine(const LatticeTopology& topology, const Vector3* v, int candidates, const int* types,
                   std::array<int, MAX_NEIGHBORS>& slots, double& bestResidual, Matrix3& bestRotation) {
    const int cn = topology.coordinationNumber;
    std::array<int, MAX_NEIGHBORS> candidate{};
    for (int pass = 0; pass < kRefinementPasses; ++pass) {
        Matrix3 covariance(Matrix3::Zero{});
        for (int k = 0; k < cn; ++k)
            for (int c = 0; c < 3; ++c) covariance.column(c) += v[slots[k]] * topology.unitVectors[k][c];
        Matrix3 rotation;
        if (!rotationFactor(covariance, rotation)) break;
        double projected = 0.0;
        double idealNorm = 0.0;
        for (int k = 0; k < cn; ++k) {
            const Vector3 ideal = rotation * topology.unitVectors[k];
            projected += v[slots[k]].dot(ideal);
            idealNorm += ideal.squaredLength();
        }
        double residual = 0.0;
        if (!assign(topology, rotation, projected / idealNorm, v, candidates, types, candidate, residual)) break;
        const bool converged = std::equal(slots.begin(), slots.begin() + cn, candidate.begin());
        slots = candidate;
        bestResidual = residual;
        bestRotation = rotation;
        if (converged) break;
    }

}

// Unit shell and anchor pair of a topology whose shell is set.
void prepareMatching(LatticeTopology& topology, const std::vector<Vector3>& shell);

}

// Crystal information of the topologies for the shared cluster builder and the contract export.
class LatticeCrystalInfo final : public StructureAnalysisCrystalInfo {
public:
    explicit LatticeCrystalInfo(std::vector<LatticeTopology> topologies);

    int findClosestSymmetryPermutation(int structureType, const Matrix3& rotation) const override;
    int coordinationNumber(int structureType) const override;
    int commonNeighborIndex(int structureType, int neighborIndex, int commonNeighborSlot) const override;
    int symmetryPermutationCount(int structureType) const override;
    int symmetryPermutationEntry(int structureType, int symmetryIndex, int neighborIndex) const override;
    const Matrix3& symmetryTransformation(int structureType, int symmetryIndex) const override;
    int symmetryInverseProduct(int structureType, int symmetryIndex, int transformationIndex) const override;
    const Vector3& latticeVector(int structureType, int latticeVectorIndex) const override;
    std::string_view topologyName(int structureType) const override;

private:
    const LatticeTopology* find(int structureType) const;

    std::vector<LatticeTopology> _topologies;
};

}
