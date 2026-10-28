#pragma once

#include <volt/core/volt.h>
#include <volt/core/lammps_parser.h>
#include <nlohmann/json.hpp>
#include <string>

namespace Volt {

using json = nlohmann::json;

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
