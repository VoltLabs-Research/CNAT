#include <volt/plugin/cnat_log.h>

#include <volt/core/type_counts.h>
#include <volt/plugin/cnat_model.h>
#include <volt/plugin/phase_field_mesh.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace Volt {

void CnatLog::dump(const CnatService::LoadedModel& loaded) {
    const CnatConfig& cfg = loaded.model.config();
    spdlog::info("Model {}: {} neighbours, hidden {}, {} layers, {} heads, MLP {}, {} parameters, classes {}",
        loaded.builtIn ? loaded.name + " (built in)" : loaded.name,
        cfg.neighbors, cfg.hidden, cfg.layers, cfg.heads, cfg.mlp, loaded.model.parameterCount(),
        json(loaded.classes).dump());
}

void CnatLog::dump(const CnatService::TrainedNeighbourTypes& types) {
    spdlog::info("Neighbour types {}, those the model was trained with", types.types);
}

void CnatLog::dump(const CnatService::NeighbourSublattice& sublattice) {
    spdlog::info("Neighbours are the {} atoms of type {} ({:.1f}% of the frame)", sublattice.atoms.natoms, sublattice.types,
        percentOf(static_cast<std::uint64_t>(sublattice.atoms.natoms), static_cast<std::size_t>(sublattice.frame.natoms)));
}

void CnatLog::dump(const CnatService::AtomClasses& classes) {
    const long long milliseconds = phaseMilliseconds();
    spdlog::info("Classified {} atoms in {} ({:.1f} us per atom)", classes.atoms,
        formatDuration(std::chrono::milliseconds(milliseconds)),
        1e3 * static_cast<double>(milliseconds) / static_cast<double>(classes.atoms));
    TypeCounts line{"Atoms", {}, classes.atoms};
    for (std::size_t c = 0; c < classes.counts.size(); ++c)
        line.counts.emplace_back(classes.names[c], static_cast<std::size_t>(classes.counts[c]));
    dump(line);
}

void CnatLog::dump(const CnatService::TopologyRequest& request) {
    if (request.requested.empty() && !request.cells)
        spdlog::info("No class of the model names a lattice topology; pass --topologies to write the structure contract");
}

void CnatLog::dump(const CnatService::UnloadableTopologies& failure) {
    spdlog::warn("Cannot load the lattice topologies ({}); the structure contract is not written", failure.error);
}

void CnatLog::dump(const CnatService::ModelCrystal& crystal) {
    int least = MAX_NEIGHBORS, most = 0;
    for (const LatticeTopology& site : crystal.cell.sites) {
        least = std::min(least, site.coordinationNumber);
        most = std::max(most, site.coordinationNumber);
    }
    spdlog::info("Crystal of {} (stored with the model): its {} sites of type {}, the sparsest sublattice, "
                 "neighbours within {:.3f} A ({} to {} per site)", crystal.className, crystal.cell.sites.size(),
                 crystal.species, crystal.cell.cutoff, least, most);
}

void CnatLog::dump(const CnatService::ReferenceCell& cell) {
    int least = MAX_NEIGHBORS, most = 0;
    for (const LatticeTopology& site : cell.cell.sites) {
        least = std::min(least, site.coordinationNumber);
        most = std::max(most, site.coordinationNumber);
    }
    spdlog::info("Reference cell {}: {} sites of type {}, neighbours within {:.3f} A ({} to {} per site)", cell.cell.name,
                 cell.cell.sites.size(), cell.types.empty() ? std::string("any") : cell.types, cell.cell.cutoff, least, most);
}

void CnatLog::dump(const ContractSiteAlignment&) {
    _alignmentMilliseconds = phaseMilliseconds();
}

void CnatLog::dump(const ContractGrainGrowth& growth) {
    const std::string growthElapsed = phaseElapsed();
    const ContractSiteAlignment& alignment = growth.alignment;
    const std::vector<LatticeTopology>& topologies = alignment.topologies;
    std::int64_t aligned = 0;
    std::vector<std::pair<std::string, std::size_t>> counts;
    // The sites of a multi-site cell share its name and are reported together.
    for (std::size_t t = 0; t < topologies.size(); ++t) {
        aligned += alignment.matched[t];
        if (t > 0 && topologies[t].cellSite >= 0 && topologies[t - 1].cellSite >= 0 && topologies[t].name == topologies[t - 1].name)
            continue;
        std::int64_t count = 0;
        for (std::size_t u = t; u < topologies.size(); ++u) {
            if (u > t && !(topologies[u].cellSite >= 0 && topologies[u].name == topologies[t].name)) break;
            count += alignment.matched[u];
            if (topologies[t].cellSite < 0) break;
        }
        counts.emplace_back(topologies[t].name, static_cast<std::size_t>(count));
    }
    char title[128];
    std::snprintf(title, sizeof(title), "Aligned %lld of %lld crystalline sites in %.0f ms", static_cast<long long>(aligned),
        static_cast<long long>(alignment.crystallineSites), static_cast<double>(_alignmentMilliseconds));
    dump(TypeCounts{title, std::move(counts), static_cast<std::size_t>(alignment.crystallineSites)});

    if (alignment.rejected > 0) {
        std::vector<double> rejectedRmsd = alignment.rejectedRmsd;
        std::ranges::nth_element(rejectedRmsd, rejectedRmsd.begin() + rejectedRmsd.size() / 2);
        spdlog::info("  {} crystalline sites ({:.1f}%) exceed the RMSD cutoff {} (median of them {:.3f}) and stay OTHER; raise --rmsd to keep them",
            alignment.rejected, percentOf(static_cast<std::uint64_t>(alignment.rejected), static_cast<std::size_t>(alignment.crystallineSites)),
            alignment.rmsdCutoff, rejectedRmsd[rejectedRmsd.size() / 2]);
    }
    if (alignment.overrides > 0)
        spdlog::info("  {} crystalline sites fit another topology at least twice as well as their class's; {} took it with "
            "at least {} shell neighbours on it, {} stood alone and stay OTHER", alignment.overrides,
            static_cast<std::int64_t>(alignment.overrides) - alignment.reverted, StructureContractDetail::kOverrideSupport,
            alignment.reverted);
    if (alignment.unaligned > 0)
        spdlog::info("  {} crystalline sites have no one-to-one alignment onto any topology and stay OTHER", alignment.unaligned);
    if (growth.grains > 0)
        spdlog::info("  Grew {} grains from their best-aligned atoms in {}: {} atoms took the site and setting of their grain "
                     "instead of their own fit; {} fits disagreed with the grain on a neighbour's site", growth.grains,
                     growthElapsed, growth.regrown, growth.inconsistent);
}

void CnatLog::dump(const ContractClusterRules& rules) {
    std::string orientationNames;
    std::string cellNames;
    std::size_t orientationTopologies = 0;
    std::size_t cellSites = 0;
    for (const LatticeTopology& topology : rules.topologies) {
        if (topology.cellSite >= 0) {
            ++cellSites;
            if (cellNames.find(topology.name) == std::string::npos) cellNames += (cellNames.empty() ? "" : ", ") + topology.name;
            continue;
        }
        if (!topology.orientationClusters) continue;
        ++orientationTopologies;
        orientationNames += (orientationNames.empty() ? "" : ", ") + topology.name;
    }
    if (orientationTopologies > 0)
        spdlog::info("  Clusters of {} grow by orientation: too few of their bonds have two common neighbours", orientationNames);
    if (cellSites > 0)
        spdlog::info("  Clusters of {} grow by orientation across its {} sites (misorientation below {} degrees)",
            cellNames, cellSites, StructureContractDetail::kCellMisorientationDegrees);
}

void CnatLog::dump(const CnatService::PhaseFieldClasses& classes) {
    TypeCounts line{"Phase field ready in " + phaseElapsed() + "; volume", {}, classes.points};
    for (std::size_t c = 0; c < classes.counts.size(); ++c)
        line.counts.emplace_back(classes.names[c], static_cast<std::size_t>(classes.counts[c]));
    dump(line);
}

void CnatLog::dump(const PhaseFieldMesh& mesh) {
    spdlog::info("  {} vertices, {} facets, area {:.1f} A^2 in {}",
        mesh.vertices.size(), mesh.facets.size(), mesh.area, phaseElapsed());
}

}
