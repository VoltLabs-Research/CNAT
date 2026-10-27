#include <volt/plugin/plugin_entry.h>
#include <volt/plugin/cnat_service.h>

using namespace Volt;
using namespace Volt::Plugin;
using S = CnatService;

static const std::vector<OptionBinding<S>> bindings = {
    opt("--model", "Built-in model (metals: LIQUID/FCC/HCP/BCC; ni-fcc-liquid: the paper's Ni model; forsterite-o: olivine on the oxygen sublattice, with --types) or a .cnat file", "metals", &S::setModelPath),
    opt("--threshold", "Crystalline probability, 1 - P(liquid), at or above which a point takes its most probable crystal class", 0.5, &S::setThreshold),
    opt("--grid_spacing", "Spacing in A of the phase-field grid; 0 classifies atoms only", 2.0, &S::setGridSpacing),
    opt("--export_interface_mesh", "Export the iso-surface of the phase field at the threshold", true, &S::setExportInterfaceMesh),
    opt("--interface_smoothing", "Gaussian smoothing in A of the phase field before the interface is extracted; 0 disables", 2.0, &S::setInterfaceSmoothing),
    opt("--export_phase_field", "Export the grid points with their class probabilities", false, &S::setExportPhaseField),
    opt("--types", "Comma-separated atom types that form the neighbour sublattice (e.g. the oxygens of a silicate); empty uses every atom", "", &S::setTypes),
    opt("--topologies", "Lattice topologies (comma-separated names of the lattice registry, e.g. fcc,hcp) the crystalline atoms are aligned onto for the structure contract OpenDXA reads; auto takes the model's crystal classes, FCC and HCP each with the other; none writes no contract", "auto", &S::setTopologies),
    opt({"--lattice_dir", "path", "Directory of the lattice topology YAMLs", "", {}, "share/volt/lattices"},
        std::function<void(S&, const OptsMap&)>([](S& s, const OptsMap& opts) { s.setLatticeDir(CLI::getString(opts, "--lattice_dir", "")); })),
    opt("--rmsd", "Largest RMSD, in units of the mean neighbour distance, at which a crystalline atom joins the structure contract; above it the atom is OTHER", 0.2, &S::setRmsd),
};

VOLT_SERVICE_PLUGIN("volt-cnat", "CNAT: Comprehensive Neighbor Analysis Transformer", S, bindings)
