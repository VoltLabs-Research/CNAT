#include <volt/plugin/structure_contract.h>
#include <volt/analysis/cluster_graph_builder.h>
#include <volt/analysis/cluster_input_preparation.h>
#include <volt/analysis/nearest_neighbor_finder.h>
#include <volt/analysis/orientation_cluster_rule_provider.h>
#include <volt/analysis/reconstructed_analysis_pipeline.h>
#include <volt/analysis/reconstructed_state_canonicalizer.h>
#include <volt/core/progress.h>
#include <spdlog/spdlog.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace Volt {

using namespace Volt::Particles;

namespace {

constexpr int kNeighborQueryCapacity = 32;
// Another topology replaces the preferred one only when its RMSD is below this fraction of the
// preferred one's: thermal noise raises every topology's RMSD alike and cannot flip a hot FCC atom to
// HCP, while a stacking-fault atom (hcp ~0, fcc > 0.25) still becomes hcp.
constexpr double kOverrideRmsdRatio = 0.5;
// ...and keeps it only with at least this many shell neighbours on the same topology, else it is OTHER:
// a stacking fault gives its atoms up to 9 such neighbours, while thermal noise near the melting point
// flips atoms one at a time (in a Ni solid-liquid coexistence at 1700 K they gave 133 A of spurious
// Shockley segments, none with this support).
constexpr int kOverrideSupport = 2;

// The shared orientation rule, for the topologies that need it; the others keep the overlap test.
class OrientationRuleFor final : public ClusterRuleProvider {
public:
    OrientationRuleFor(std::shared_ptr<const std::vector<OrientationClusterAtomState>> orientations, std::vector<int> structureTypes)
        : _rule(std::move(orientations)), _structureTypes(std::move(structureTypes)) {}

    void initializeClusterSeed(const StructureAnalysis& analysis, const AnalysisContext& context, Cluster& cluster,
                               int seedAtomIndex, int structureType) const override {
        if (handles(structureType)) _rule.initializeClusterSeed(analysis, context, cluster, seedAtomIndex, structureType);
    }

    ClusterRuleDecision tryAssignNeighbor(const StructureAnalysis& analysis, const AnalysisContext& context,
                                          const Cluster& cluster, int currentAtomIndex, int neighborAtomIndex,
                                          int neighborIndex, int structureType, int& outNeighborSymmetry) const override {
        if (!handles(structureType)) return ClusterRuleDecision::Unhandled;
        return _rule.tryAssignNeighbor(analysis, context, cluster, currentAtomIndex, neighborAtomIndex, neighborIndex,
                                       structureType, outNeighborSymmetry);
    }

    ClusterRuleDecision tryCalculateTransition(const StructureAnalysis& analysis, const AnalysisContext& context,
                                               int atomIndex, int neighborAtomIndex, int neighborIndex,
                                               Matrix3& outTransition) const override {
        if (!handles(context.structureTypes->getInt(atomIndex))) return ClusterRuleDecision::Unhandled;
        return _rule.tryCalculateTransition(analysis, context, atomIndex, neighborAtomIndex, neighborIndex, outTransition);
    }

private:
    bool handles(int structureType) const { return std::ranges::find(_structureTypes, structureType) != _structureTypes.end(); }

    OrientationClusterRuleProvider _rule;
    std::vector<int> _structureTypes;
};

struct AlignmentTally {
    std::vector<std::int64_t> matched;
    std::int64_t rejected = 0;
    std::int64_t unaligned = 0;
    std::vector<double> rejectedRmsd;
    std::vector<std::size_t> overrides;
};

}

std::optional<std::string> writeStructureContract(
    const LammpsParser::Frame& frame,
    const std::string& outputBase,
    const std::vector<int>& sites,
    const std::vector<int>& preferredTopology,
    const std::vector<LatticeTopology>& topologies,
    double rmsdCutoff,
    json& result,
    StructureContractAtoms& atoms) {
    const std::size_t natoms = static_cast<std::size_t>(frame.natoms);
    int maxCn = 0;
    std::string names;
    for (const LatticeTopology& topology : topologies) {
        maxCn = std::max(maxCn, topology.coordinationNumber);
        names += (names.empty() ? "" : ",") + topology.name;
    }
    if (sites.size() <= static_cast<std::size_t>(maxCn))
        return "The lattice sites are fewer than the " + std::to_string(maxCn + 1) + " the topologies need";

    std::string error;
    auto session = AnalysisPipelineUtils::prepareAnalysisSession(frame, LATTICE_OTHER, &error);
    if (!session) return error;
    AnalysisContext& context = session->context;
    StructureAnalysis analysis(context);
    analysis.setCrystalInfoProvider(std::make_shared<LatticeCrystalInfo>(topologies));

    auto sitePositions = std::make_shared<ParticleProperty>(sites.size(), ParticleProperty::PositionProperty, 0, true);
    Point3* sitePoints = sitePositions->dataPoint3();
    for (std::size_t s = 0; s < sites.size(); ++s) sitePoints[s] = context.positions->getPoint3(sites[s]);
    NearestNeighborFinder finder(maxCn);
    if (!finder.prepare(sitePositions.get(), frame.simulationCell))
        return std::string("NearestNeighborFinder::prepare failed on the lattice sites");

    std::int64_t crystallineSites = 0;
    for (const int atom : sites) crystallineSites += preferredTopology[atom] != kNotCrystalline ? 1 : 0;
    spdlog::info("Aligning {} crystalline sites onto {} (RMSD cutoff {})...", crystallineSites, names, rmsdCutoff);

    atoms.topology.assign(natoms, -1);
    atoms.rmsd.assign(natoms, std::numeric_limits<float>::quiet_NaN());
    std::vector<std::array<int, MAX_NEIGHBORS>> siteNeighbors(sites.size());
    auto orientations = std::make_shared<std::vector<OrientationClusterAtomState>>(natoms);
    tbb::enumerable_thread_specific<AlignmentTally> tallies([&] {
        AlignmentTally tally;
        tally.matched.assign(topologies.size(), 0);
        return tally;
    });
    Stopwatch timer;
    ProgressReporter progress("Aligning crystalline sites", sites.size());
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, sites.size(), 256),
        [&](const tbb::blocked_range<std::size_t>& r) {
            AlignmentTally& tally = tallies.local();
            NearestNeighborFinder::Query<kNeighborQueryCapacity> query(finder);
            std::array<Vector3, MAX_NEIGHBORS> deltas;
            std::vector<LatticeMatch> matches(topologies.size());
            std::vector<unsigned char> aligned(topologies.size());
            for (std::size_t s = r.begin(); s < r.end(); ++s) {
                const int atom = sites[s];
                const int preferred = preferredTopology[atom];
                if (preferred == kNotCrystalline) continue;
                query.findNeighbors(s);
                const auto& found = query.results();
                const int count = std::min(static_cast<int>(found.size()), maxCn);
                for (int j = 0; j < count; ++j) deltas[j] = found[j].delta;

                int bestTopology = -1;
                for (std::size_t t = 0; t < topologies.size(); ++t) {
                    aligned[t] = topologies[t].coordinationNumber <= count &&
                                 matchLattice(topologies[t], deltas.data(), matches[t]);
                    if (aligned[t] && (bestTopology < 0 || matches[t].rmsd < matches[bestTopology].rmsd))
                        bestTopology = static_cast<int>(t);
                }
                if (bestTopology < 0) {
                    ++tally.unaligned;
                    continue;
                }
                bool overridden = false;
                if (preferred >= 0 && preferred != bestTopology) {
                    if (aligned[preferred] && matches[bestTopology].rmsd >= kOverrideRmsdRatio * matches[preferred].rmsd)
                        bestTopology = preferred;
                    else
                        overridden = true;
                }
                const LatticeMatch& best = matches[bestTopology];
                atoms.rmsd[atom] = static_cast<float>(best.rmsd);
                if (best.rmsd > rmsdCutoff) {
                    ++tally.rejected;
                    tally.rejectedRmsd.push_back(best.rmsd);
                    continue;
                }
                ++tally.matched[bestTopology];
                atoms.topology[atom] = bestTopology;
                (*orientations)[atom].orientation = best.rotation;
                (*orientations)[atom].valid = true;
                for (int k = 0; k < topologies[bestTopology].coordinationNumber; ++k)
                    siteNeighbors[s][k] = sites[found[best.slots[k]].index];
                if (overridden) tally.overrides.push_back(s);
            }
            progress.advance(r.size());
        });

    std::vector<std::int64_t> matched(topologies.size(), 0);
    std::int64_t rejected = 0;
    std::int64_t unaligned = 0;
    std::vector<double> rejectedRmsd;
    std::vector<std::size_t> overrides;
    for (const AlignmentTally& tally : tallies) {
        for (std::size_t t = 0; t < topologies.size(); ++t) matched[t] += tally.matched[t];
        rejected += tally.rejected;
        unaligned += tally.unaligned;
        rejectedRmsd.insert(rejectedRmsd.end(), tally.rejectedRmsd.begin(), tally.rejectedRmsd.end());
        overrides.insert(overrides.end(), tally.overrides.begin(), tally.overrides.end());
    }

    // Sites off their class's topology without the support of their shell become OTHER; the support
    // counts the assignments before any of them is dropped.
    const std::vector<int> assigned = atoms.topology;
    std::int64_t reverted = 0;
    for (const std::size_t site : overrides) {
        const int atom = sites[site];
        const int taken = assigned[atom];
        int support = 0;
        for (int k = 0; k < topologies[taken].coordinationNumber; ++k)
            support += assigned[siteNeighbors[site][k]] == taken ? 1 : 0;
        if (support >= kOverrideSupport) continue;
        ++reverted;
        --matched[taken];
        atoms.topology[atom] = -1;
        (*orientations)[atom].valid = false;
    }
    const std::int64_t overridden = static_cast<std::int64_t>(overrides.size()) - reverted;
    std::int64_t aligned = 0;
    std::string line;
    for (std::size_t t = 0; t < topologies.size(); ++t) {
        aligned += matched[t];
        char entry[96];
        std::snprintf(entry, sizeof(entry), "%s%s %lld (%.1f%%)", line.empty() ? "" : ", ", topologies[t].name.c_str(),
            static_cast<long long>(matched[t]), percentOf(static_cast<std::uint64_t>(matched[t]), static_cast<std::size_t>(crystallineSites)));
        line += entry;
    }
    spdlog::info("Aligned {} of {} crystalline sites in {}: {}", aligned, crystallineSites, timer.elapsed(), line);
    if (rejected > 0) {
        std::ranges::nth_element(rejectedRmsd, rejectedRmsd.begin() + rejectedRmsd.size() / 2);
        spdlog::info("  {} crystalline sites ({:.1f}%) exceed the RMSD cutoff {} (median of them {:.3f}) and stay OTHER; raise --rmsd to keep them",
            rejected, percentOf(static_cast<std::uint64_t>(rejected), static_cast<std::size_t>(crystallineSites)),
            rmsdCutoff, rejectedRmsd[rejectedRmsd.size() / 2]);
    }
    if (!overrides.empty())
        spdlog::info("  {} crystalline sites fit another topology at least twice as well as their class's; {} took it with "
            "at least {} shell neighbours on it, {} stood alone and stay OTHER", overrides.size(), overridden,
            kOverrideSupport, reverted);
    if (unaligned > 0)
        spdlog::info("  {} crystalline sites have no one-to-one alignment onto any topology and stay OTHER", unaligned);

    std::vector<int> siteIndex(natoms, -1);
    for (std::size_t s = 0; s < sites.size(); ++s) siteIndex[sites[s]] = static_cast<int>(s);
    int* types = context.structureTypes->dataInt();
    int* counts = context.neighborCounts->dataInt();
    int* offsets = context.neighborOffsets->dataInt();
    offsets[0] = 0;
    for (std::size_t i = 0; i < natoms; ++i) {
        const int t = atoms.topology[i];
        types[i] = t >= 0 ? topologies[t].structureType : LATTICE_OTHER;
        counts[i] = t >= 0 ? topologies[t].coordinationNumber : 0;
        offsets[i + 1] = offsets[i] + counts[i];
    }
    context.neighborIndices = std::make_shared<ParticleProperty>(static_cast<std::size_t>(offsets[natoms]), DataType::Int, 1, 0, false);
    int* indices = context.neighborIndices->dataInt();
    double maxDistance = 0.0;
    for (std::size_t i = 0; i < natoms; ++i) {
        for (int k = 0; k < counts[i]; ++k) {
            const int neighbor = siteNeighbors[siteIndex[i]][k];
            indices[offsets[i] + k] = neighbor;
            maxDistance = std::max(maxDistance, context.simCell.wrapVector(
                context.positions->getPoint3(neighbor) - context.positions->getPoint3(i)).length());
        }
    }
    context.maximumNeighborDistance = maxDistance;

    spdlog::info("Building clusters...");
    timer.restart();
    ClusterInputAdapterUtils::prepareSymmetryAwareClusterInputs(analysis, context, true,
        [&](std::size_t atomIndex, int structureType) {
            return structureType != LATTICE_OTHER && analysis.numberOfNeighbors(static_cast<int>(atomIndex)) > 0;
        });
    // Topologies whose overlap-grown bonds do not span space (sc, st, a7) join their atoms by the shared
    // orientation rule instead, as PTM does for SC.
    std::vector<int> orientationTypes;
    std::string orientationNames;
    for (const LatticeTopology& topology : topologies) {
        if (!topology.orientationClusters) continue;
        orientationTypes.push_back(topology.structureType);
        orientationNames += (orientationNames.empty() ? "" : ", ") + topology.name;
    }
    if (!orientationTypes.empty()) {
        spdlog::info("  Clusters of {} grow by orientation: too few of their bonds have two common neighbours", orientationNames);
        analysis.setClusterRuleProvider(std::make_shared<OrientationRuleFor>(orientations, std::move(orientationTypes)));
    }
    ClusterBuilder clusterBuilder(analysis, context);
    clusterBuilder.build(false);
    normalizeReconstructedClusterGraphForExport(analysis, context);
    const auto& clusters = analysis.clusterGraph().clusters();
    // The cluster graph always holds the null cluster 0.
    const std::size_t clusterCount = clusters.empty() ? 0 : clusters.size() - 1;
    std::int64_t clusteredSites = 0;
    for (const int atom : sites) clusteredSites += context.atomClusters->getInt(atom) != 0 ? 1 : 0;
    spdlog::info("Built {} clusters in {} ({} of {} lattice sites in clusters, {:.1f}%)", clusterCount, timer.elapsed(),
        clusteredSites, sites.size(), percentOf(static_cast<std::uint64_t>(clusteredSites), sites.size()));

    // Atoms off the neighbour sublattice are no lattice sites of the topologies; left in the contract as
    // OTHER they would turn every Delaunay cell into a defect cell.
    std::vector<bool> contractSites;
    const std::vector<bool>* contractMask = nullptr;
    if (sites.size() < natoms) {
        contractSites.assign(natoms, false);
        for (const int atom : sites) contractSites[atom] = true;
        contractMask = &contractSites;
    }
    spdlog::info("Writing the structure contract (annotated dump, cluster graph and neighbour lattice)...");
    timer.restart();
    if (!AnalysisPipelineUtils::appendClusterOutputs(frame, outputBase, outputBase + "_annotated.dump",
            context, analysis, result, &error, {}, contractMask))
        return error;
    spdlog::info("Structure contract written in {}", timer.elapsed());

    json& listing = result["main_listing"];
    listing["lattice_topologies"] = names;
    listing["lattice_rmsd_cutoff"] = rmsdCutoff;
    listing["lattice_sites"] = static_cast<std::int64_t>(sites.size());
    listing["aligned_sites"] = aligned;
    listing["sites_over_rmsd_cutoff"] = rejected;
    listing["sites_off_their_class_topology"] = overridden;
    listing["clusters"] = static_cast<std::int64_t>(clusterCount);
    json rows = json::array();
    for (std::size_t t = 0; t < topologies.size(); ++t) {
        rows.push_back({
            {"topology", topologies[t].name},
            {"structure_id", topologies[t].structureType},
            {"coordination_number", topologies[t].coordinationNumber},
            {"count", matched[t]},
            {"fraction", crystallineSites ? static_cast<double>(matched[t]) / static_cast<double>(crystallineSites) : 0.0},
        });
    }
    result["sub_listings"]["lattice_alignment"] = std::move(rows);
    return std::nullopt;
}

}
