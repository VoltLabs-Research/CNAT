#pragma once

#include <volt/core/lammps_parser.h>
#include <volt/plugin/lattice_matcher.h>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace Volt {

using json = nlohmann::json;

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
    StructureContractAtoms& atoms);

}
