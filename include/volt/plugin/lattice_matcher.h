#pragma once

#include <volt/analysis/crystal_topology_library.h>
#include <volt/analysis/structure_analysis.h>

#include <array>
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
