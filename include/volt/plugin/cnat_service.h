#pragma once

#include <volt/core/volt.h>
#include <volt/core/lammps_parser.h>
#include <volt/core/simulation_cell.h>
#include <volt/plugin/lattice_matcher.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Volt {

using json = nlohmann::json;

class CnatLog;
class CnatModel;
class ColumnarRowWriter;
class NearestNeighborFinder;
struct PhaseFieldMesh;

// Phase identification at arbitrary points with CNAT (Watanabe, Sato & Shibuta 2027). Each atom is
// labelled from the k nearest atoms of its own position (itself included); a regular grid over the
// cell gives the phase field. Class 0 of a model is the disordered phase, and the interface mesh is
// the iso-surface where 1 - P(class 0) equals the threshold. With atom types set, only atoms of those
// types are neighbours (a sublattice model, e.g. the oxygens of a silicate), and every atom is still
// labelled from the sublattice around its position.
//
// The crystalline atoms also give the structure-identification contract OpenDXA reads: each one is
// aligned onto the lattice topologies (YAML files of the lattice registry, any of them, not only
// those of PTM or CNA) and takes the best fit, from which the shared cluster builder grows the
// crystal clusters. CNAT decides where the crystal is; the topology gives the neighbour order.
class CnatService {
public:
    // The model a run classifies with, by the name the results give it, and its class names.
    struct LoadedModel {
        const CnatModel& model;
        const std::string& name;
        bool builtIn;
        const std::vector<std::string>& classes;
    };

    // The neighbour types a run takes from its model, for want of --types.
    struct TrainedNeighbourTypes {
        const std::string& types;
    };

    // The atoms of the neighbour types, out of the frame.
    struct NeighbourSublattice {
        const LammpsParser::Frame& atoms;
        const LammpsParser::Frame& frame;
        const std::string& types;
    };

    // How many atoms took each class.
    struct AtomClasses {
        const std::vector<std::int64_t>& counts;
        const std::vector<std::string>& names;
        std::size_t atoms;
    };

    // The topologies the contract was asked for by name, and whether a reference cell gives it others.
    struct TopologyRequest {
        const std::vector<std::string>& requested;
        bool cells;
    };

    // Why the topologies the model names could not be loaded, which leaves the contract unwritten.
    struct UnloadableTopologies {
        const char* error;
    };

    // The sites of the crystal a model stores for a class, on the sublattice of one species.
    struct ModelCrystal {
        const std::string& className;
        const CellTopologies& cell;
        int species;
    };

    // The sites of the --cell reference crystal of these atom types.
    struct ReferenceCell {
        const CellTopologies& cell;
        const std::string& types;
    };

    // How many grid points of the phase field took each class.
    struct PhaseFieldClasses {
        const std::vector<std::int64_t>& counts;
        const std::vector<std::string>& names;
        std::size_t points;
    };

    void setModelPath(std::string path) { _modelPath = std::move(path); }
    void setThreshold(double threshold) { _threshold = threshold; }
    void setGridSpacing(double spacing) { _gridSpacing = spacing; }
    void setExportInterfaceMesh(bool on) { _exportInterfaceMesh = on; }
    void setInterfaceSmoothing(double sigma) { _interfaceSmoothing = sigma; }
    void setExportPhaseField(bool on) { _exportPhaseField = on; }
    void setTypes(std::string types) { _types = std::move(types); }
    void setTopologies(std::string topologies) { _topologies = std::move(topologies); }
    void setLatticeDir(std::string directory) { _latticeDir = std::move(directory); }
    void setRmsd(double rmsd) { _rmsd = rmsd; }
    void setCell(std::string path) { _cell = std::move(path); }
    void setCellTypes(std::string types) { _cellTypes = std::move(types); }
    void setCellCutoff(double cutoff) { _cellCutoff = cutoff; }
    void setContractTypes(std::string types) { _contractTypes = std::move(types); }

    json compute(const LammpsParser::Frame& frame, const std::string& outputBase);

private:
    static constexpr int kMaxNeighbors = 128;

    // The crystal a crystal class of the model was trained on (export_model.py --crystal): a lattice topology
    // of the registry, or the reference cell its training crystal was built from.
    struct ClassCrystal {
        std::string topology;
        std::optional<CellAtoms> cell;
    };

    static std::vector<std::string> classNames(const CnatModel& model);
    static std::string modelName(const CnatModel& model);
    // Comma-separated atom types; empty selects every atom.
    static std::optional<std::vector<int>> parseTypes(const std::string& text);
    // Structure id of every class: StructureType's when the class names one of its structures (FCC, HCP,
    // BCC, ...), so CNAT's atoms read as the other producers'; class 0 is OTHER. The other crystal classes
    // of the built-in models keep one id each across models (plugin.json names them); those of any other
    // model count up from kCustomTopologyTypeBase + 100.
    static std::vector<int> structureIds(const std::vector<std::string>& names);
    static std::vector<ClassCrystal> modelCrystals(const CnatModel& model, const std::vector<std::string>& names);
    // The sublattice of a cell the structure contract is written on: the species with the fewest atoms in
    // the cell. It is the sparsest one, whose sites are farthest apart and least alike under the other
    // sublattices' pseudo-symmetries (forsterite's oxygens are nearly hexagonal close packed, its Si are
    // not), and one Wyckoff orbit has every translation of the crystal, so its Burgers vectors are the
    // crystal's.
    static int contractSpecies(const CellAtoms& cell);
    // The lattice topologies of the model's crystal classes, each close-packed or diamond one with its
    // partner: the planar faults of either are local patches of the other, as CNA labels both from
    // either input lattice.
    static std::vector<std::string> modelTopologies(const std::vector<std::string>& names,
                                                    const std::vector<ClassCrystal>& crystals);
    // probability_crystalline (1 - P(class 0), the field of the interface mesh, with the same name for
    // every model) and one probability_<class> column per class.
    static void writeProbabilities(ColumnarRowWriter& w, const float* p, const std::vector<std::string>& names);
    // Class probabilities of every point, row-major [point][class].
    template<typename PointAt>
    static std::vector<float> classifyPoints(const CnatModel& model, const NearestNeighborFinder& finder,
                                             std::size_t count, PointAt pointAt, const char* label);
    // Class 0 is the disordered phase. A point is crystalline when 1 - P(class 0) reaches the threshold,
    // and then takes its most probable crystal class; for a binary model this is P(class 1) >= threshold.
    static std::vector<int> assignLabels(const std::vector<float>& probabilities, int classes, double threshold);
    // Points along each cell vector of the phase-field grid, about `spacing` apart.
    static std::array<int, 3> phaseFieldDims(const SimulationCell& cell, double spacing);
    static json countRows(const std::vector<std::int64_t>& counts, const std::vector<std::string>& names,
                          const std::vector<int>& ids, std::size_t total, const char* countKey);

    std::string _modelPath;
    double _threshold = 0.5;
    double _gridSpacing = 2.0;
    bool _exportInterfaceMesh = true;
    double _interfaceSmoothing = 2.0;
    bool _exportPhaseField = false;
    std::string _types;
    std::string _topologies = "auto";
    std::string _latticeDir;
    double _rmsd = 0.2;
    std::string _cell;
    std::string _cellTypes;
    double _cellCutoff = 0.0;
    std::string _contractTypes;
};

}
