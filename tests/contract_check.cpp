// Structure contract of a dump without running the model: every atom is crystalline with no preferred
// topology, or each takes its preference from a file (one integer per atom, in dump order: -2 not
// crystalline, -1 any topology, else the index into <topologies>; e.g. from a CNAT atoms table). A
// perfect lattice must align every atom with RMSD ~0 and give one cluster.
//   cnat-contract-check <dump> <output_base> <topologies> <rmsd> [lattice_dir] [preferences]
#include <volt/plugin/structure_contract.h>
#include <volt/core/lammps_parser.h>
#include <volt/structures/crystal_topology_registry.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <fstream>
#include <numeric>

using namespace Volt;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <dump> <output_base> <topologies> <rmsd> [lattice_dir] [preferences]\n", argv[0]);
        return 2;
    }
    if (argc > 5 && argv[5][0]) setCrystalTopologySearchRoot(argv[5]);
    LammpsParser parser;
    LammpsParser::Frame frame;
    if (!parser.parseFile(argv[1], frame)) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    std::vector<std::string> names;
    for (std::string list = argv[3]; !list.empty();) {
        const std::size_t comma = list.find(',');
        names.push_back(list.substr(0, comma));
        list = comma == std::string::npos ? "" : list.substr(comma + 1);
    }
    const auto topologies = loadLatticeTopologies(names);
    std::vector<int> sites(frame.natoms);
    std::iota(sites.begin(), sites.end(), 0);
    std::vector<int> preferred(frame.natoms, kAnyTopology);
    if (argc > 6) {
        std::ifstream file(argv[6]);
        for (int i = 0, value = 0; i < frame.natoms && file >> value; ++i) preferred[i] = value;
    }
    json result;
    StructureContractAtoms atoms;
    if (auto error = writeStructureContract(frame, argv[2], sites, preferred, topologies, std::stod(argv[4]), result, atoms)) {
        std::fprintf(stderr, "%s\n", error->c_str());
        return 1;
    }
    std::printf("%s\n", result["main_listing"].dump().c_str());
    return 0;
}
