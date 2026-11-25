#pragma once

#include <volt/analysis/cluster_rule_provider.h>
#include <volt/analysis/orientation_cluster_rule_provider.h>
#include <volt/analysis/structure_analysis.h>
#include <volt/core/lammps_parser.h>
#include <volt/plugin/lattice_matcher.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Volt {

using json = nlohmann::json;

class CnatLog;
class NearestNeighborFinder;

// Topology a site prefers, per atom: one of these, or the index of the topology its CNAT class names.
inline constexpr int kNotCrystalline = -2;
inline constexpr int kAnyTopology = -1;

// Per-atom outcome of the contract, for the atoms table.
struct StructureContractAtoms {
    // Index into the topologies, or -1 when the atom is not a crystalline lattice site.
    std::vector<int> topology;
    // RMSD of the best alignment of every crystalline site (also of the ones the cutoff rejects); NaN
    // for the rest.
    std::vector<float> rmsd;
};

// How the crystalline sites aligned onto the topologies before any grain of a multi-site cell grew: the sites each
// topology took, and the sites that fit none, fit only above the RMSD cutoff, or fit another topology better than
// their class's (of which those without the support of their shell were reverted to OTHER).
struct ContractSiteAlignment {
    const std::vector<LatticeTopology>& topologies;
    double rmsdCutoff;
    std::int64_t crystallineSites = 0;
    std::vector<std::int64_t> matched;
    std::int64_t rejected = 0;
    // The RMSD of each site over the cutoff, of which the log names the median.
    std::vector<double> rejectedRmsd;
    std::int64_t unaligned = 0;
    std::size_t overrides = 0;
    std::int64_t reverted = 0;
};

// The grains grown from the best-aligned atoms of the multi-site cells (none without one), which move sites of the
// alignment between topologies: the atoms that took their grain's site or setting instead of their own fit, the
// grains, and the fits turned down as inconsistent with their grain.
struct ContractGrainGrowth {
    const ContractSiteAlignment& alignment;
    std::int64_t regrown = 0;
    std::int64_t grains = 0;
    std::int64_t inconsistent = 0;
};

// The topologies whose clusters grow by orientation: those whose overlap-grown bonds do not span space, and the
// sites of the multi-site cells.
struct ContractClusterRules {
    const std::vector<LatticeTopology>& topologies;
};

// Writes the volt/structure-identification@1 contract that OpenDXA, ElasticStrain and
// LineReconstructionDXA read: {base}_annotated.dump, the cluster tables and the neighbour lattice.
// CNAT decides which atoms are crystalline and of which class. Each crystalline site is aligned onto
// every topology and keeps the one its class names unless another fits at least twice as well and two
// of its shell neighbours took that one too (a stacking fault, also for a model without an HCP class);
// alone, such a site is OTHER. A class without a topology takes the best fit.
// Within the RMSD cutoff, the alignment gives the neighbour order the shared cluster builder grows the
// crystal clusters from. Sites are the atoms of the neighbour sublattice; when they are not all the
// atoms, the contract holds only them. Returns the error, if any.
std::optional<std::string> writeStructureContract(
    const LammpsParser::Frame& frame,
    const std::string& outputBase,
    const std::vector<int>& sites,
    const std::vector<int>& preferredTopology,
    const std::vector<LatticeTopology>& topologies,
    double rmsdCutoff,
    json& result,
    StructureContractAtoms& atoms,
    CnatLog& log);

// The phases of writeStructureContract and the cluster rules of its topologies.
namespace StructureContractDetail {

inline constexpr int kNeighborQueryCapacity = 32;
// Candidates past the largest shell for the sites of a cell of several species (matchLattice).
inline constexpr int kExtraCandidates = 8;
// Another topology replaces the preferred one only when its RMSD is below this fraction of the
// preferred one's: thermal noise raises every topology's RMSD alike and cannot flip a hot FCC atom to
// HCP, while a stacking-fault atom (hcp ~0, fcc > 0.25) still becomes hcp.
inline constexpr double kOverrideRmsdRatio = 0.5;
// ...and keeps it only with at least this many shell neighbours on the same topology, else it is OTHER:
// a stacking fault gives its atoms up to 9 such neighbours, while thermal noise near the melting point
// flips atoms one at a time (in a Ni solid-liquid coexistence at 1700 K they gave 133 A of spurious
// Shockley segments, none with this support).
inline constexpr int kOverrideSupport = 2;
// Two neighbours of a multi-site cell join one cluster when their orientations differ by less than this:
// thermal noise turns a neighbourhood by a degree or two, and a low-angle boundary turns it by its
// misorientation only across its dislocation cores, so a grain stays one cluster bond by bond.
inline constexpr double kCellMisorientationDegrees = 10.0;
// ...and when each atom has a shell vector within this angle (cosine) of the bond between them.
inline constexpr double kCellBondAlignment = 0.95;

// The shared orientation rule, for the topologies that need it; the others keep the overlap test. The
// rule is called through its base class, as PTM does: structure-identification declares Cluster as a
// class and defines it as a struct, and MSVC mangles the two differently, so a direct call would not
// link on Windows.
class OrientationRuleFor final : public ClusterRuleProvider {
public:
    OrientationRuleFor(std::shared_ptr<const std::vector<OrientationClusterAtomState>> orientations, std::vector<int> structureTypes);

    void initializeClusterSeed(const StructureAnalysis& analysis, const AnalysisContext& context, Cluster& cluster,
                               int seedAtomIndex, int structureType) const override;
    ClusterRuleDecision tryAssignNeighbor(const StructureAnalysis& analysis, const AnalysisContext& context,
                                          const Cluster& cluster, int currentAtomIndex, int neighborAtomIndex,
                                          int neighborIndex, int structureType, int& outNeighborSymmetry) const override;
    ClusterRuleDecision tryCalculateTransition(const StructureAnalysis& analysis, const AnalysisContext& context,
                                               int atomIndex, int neighborAtomIndex, int neighborIndex,
                                               Matrix3& outTransition) const override;

private:
    bool handles(int structureType) const { return std::ranges::find(_structureTypes, structureType) != _structureTypes.end(); }

    std::shared_ptr<const ClusterRuleProvider> _rule;
    std::vector<int> _structureTypes;
};

// Cluster rule of the sites of a multi-site cell (loadCellTopologies), which are topologies of their own
// that share a name and the crystal frame and have the identity as their only symmetry: atoms of any of
// them grow into one cluster by orientation, and their transitions are the relative orientations. The
// shared orientation rule, which takes one structure type at a time, keeps the other topologies that
// need it; the rest keep the overlap test. Atoms of different types are joined only within the cell.
class ContractRuleProvider final : public ClusterRuleProvider {
public:
    ContractRuleProvider(std::shared_ptr<const std::vector<OrientationClusterAtomState>> orientations,
                         std::vector<int> orientationTypes, std::vector<int> cellTypes);

    bool allowsCrossStructureGrowth() const override { return !_cellTypes.empty(); }

    void initializeClusterSeed(const StructureAnalysis& analysis, const AnalysisContext& context, Cluster& cluster,
                               int seedAtomIndex, int structureType) const override;
    ClusterRuleDecision tryAssignNeighbor(const StructureAnalysis& analysis, const AnalysisContext& context,
                                          const Cluster& cluster, int currentAtomIndex, int neighborAtomIndex,
                                          int neighborIndex, int structureType, int& outNeighborSymmetry) const override;
    ClusterRuleDecision tryCalculateTransition(const StructureAnalysis& analysis, const AnalysisContext& context,
                                               int atomIndex, int neighborAtomIndex, int neighborIndex,
                                               Matrix3& outTransition) const override;

private:
    bool isCell(int type) const { return std::ranges::find(_cellTypes, type) != _cellTypes.end(); }
    bool isOrientation(int type) const { return std::ranges::find(_orientationTypes, type) != _orientationTypes.end(); }

    const Matrix3* orientation(int atom) const {
        const auto& state = (*_orientations)[static_cast<std::size_t>(atom)];
        return state.valid ? &state.orientation : nullptr;
    }

    // Largest cosine between the bond and a shell vector of the atom, in its orientation.
    static double bondAlignment(const StructureAnalysis& analysis, int type, const Matrix3& rotation, const Vector3& bond) {
        double best = -1.0;
        for (int k = 0; k < analysis.coordinationNumber(type); ++k) {
            const Vector3 predicted = rotation * analysis.latticeVector(type, k);
            const double length = predicted.length();
            if (length > EPSILON) best = std::max(best, predicted.dot(bond) / length);
        }
        return best;
    }

    ClusterRuleDecision assignCellNeighbor(const StructureAnalysis& analysis, const AnalysisContext& context, int current,
                                           int neighbor, int, int currentType, int neighborType, int& outNeighborSymmetry) const;

    // Gram-Schmidt with a proper handedness, as the shared orientation rule orthonormalizes transitions.
    static bool orthonormalizeMatrix(const Matrix3& input, Matrix3& output) {
        Vector3 first = input.column(0);
        if (first.length() <= EPSILON) return false;
        first /= first.length();
        Vector3 second = input.column(1) - first * first.dot(input.column(1));
        if (second.length() <= EPSILON) return false;
        second /= second.length();
        Vector3 third = first.cross(second);
        output = Matrix3(first, second, third);
        return true;
    }

    std::shared_ptr<const std::vector<OrientationClusterAtomState>> _orientations;
    std::shared_ptr<const ClusterRuleProvider> _orientationRule;
    std::vector<int> _orientationTypes;
    std::vector<int> _cellTypes;
    double _minimumTrace;
};

// The alignments one thread counts.
struct AlignmentTally {
    std::vector<std::int64_t> matched;
    std::int64_t rejected = 0;
    std::int64_t unaligned = 0;
    std::vector<double> rejectedRmsd;
    std::vector<std::size_t> overrides;
};

// Aligns every crystalline site onto the topologies and keeps the best fit within the cutoff (its class's unless
// another fits twice as well with the support of its shell), with its orientation and its shell in slot order.
ContractSiteAlignment alignSites(
    const LammpsParser::Frame& frame, const std::vector<int>& sites, const std::vector<int>& preferredTopology,
    const std::vector<LatticeTopology>& topologies, const NearestNeighborFinder& finder, int candidates,
    double rmsdCutoff, StructureContractAtoms& atoms, std::vector<OrientationClusterAtomState>& orientations,
    std::vector<std::array<int, MAX_NEIGHBORS>>& siteNeighbors);

// The sites of a multi-site cell that a rotation or translation of its space group relates (the four Si
// of Pbnm are one Wyckoff site) or that a pseudo-symmetry relates (forsterite's oxygens are nearly
// hexagonal close packed, and some of its sites nearly repeat at half a lattice vector) fit an atom
// alike, so an atom's own best fit can leave it in a setting or on a site its grain does not have, and
// circuits through it would not close on a lattice vector. Each grain therefore grows from its
// best-aligned atom, the seeds taken in order of RMSD, as a registration does: a neighbour is the site
// the bond to it lands on in the cell, and takes the orientation of the atom it is reached from, if its
// refined fit there stays within the cluster misorientation and the RMSD cutoff, and if the fit agrees
// with every neighbour already in the grain on which site each is (the bonds read the same both ways;
// where a pseudo-translation would shift the sites, they do not). A neighbour that fits
// under no grain is reached from elsewhere or, if it aligned on its own, seeds a grain of its own, so
// neither a setting nor a site is forced across a grain boundary or a dislocation core.
ContractGrainGrowth growCellGrains(
    ContractSiteAlignment& alignment, const LammpsParser::Frame& frame, const std::vector<int>& sites,
    const std::vector<int>& siteIndex, const NearestNeighborFinder& finder, int candidates,
    const std::vector<LatticeTopology>& topologies, const std::vector<int>& preferredTopology, double rmsdCutoff,
    StructureContractAtoms& atoms, std::vector<OrientationClusterAtomState>& orientations,
    std::vector<std::array<int, MAX_NEIGHBORS>>& siteNeighbors);

}

}
