#include <volt/plugin/structure_contract.h>
#include <volt/plugin/cnat_log.h>
#include <volt/analysis/cluster_graph_builder.h>
#include <volt/analysis/cluster_input_preparation.h>
#include <volt/analysis/nearest_neighbor_finder.h>
#include <volt/analysis/orientation_cluster_rule_provider.h>
#include <volt/analysis/reconstructed_analysis_pipeline.h>
#include <volt/analysis/reconstructed_state_canonicalizer.h>
#include <volt/core/progress.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <numeric>

namespace Volt {

using namespace Volt::Particles;
using namespace StructureContractDetail;

StructureContractDetail::OrientationRuleFor::OrientationRuleFor(
    std::shared_ptr<const std::vector<OrientationClusterAtomState>> orientations, std::vector<int> structureTypes)
    : _rule(std::make_shared<OrientationClusterRuleProvider>(std::move(orientations))),
      _structureTypes(std::move(structureTypes)) {}

void StructureContractDetail::OrientationRuleFor::initializeClusterSeed(
    const StructureAnalysis& analysis, const AnalysisContext& context, Cluster& cluster, int seedAtomIndex,
    int structureType) const {
    if (handles(structureType)) _rule->initializeClusterSeed(analysis, context, cluster, seedAtomIndex, structureType);
}

ClusterRuleDecision StructureContractDetail::OrientationRuleFor::tryAssignNeighbor(
    const StructureAnalysis& analysis, const AnalysisContext& context, const Cluster& cluster, int currentAtomIndex,
    int neighborAtomIndex, int neighborIndex, int structureType, int& outNeighborSymmetry) const {
    if (!handles(structureType)) return ClusterRuleDecision::Unhandled;
    return _rule->tryAssignNeighbor(analysis, context, cluster, currentAtomIndex, neighborAtomIndex, neighborIndex,
                                    structureType, outNeighborSymmetry);
}

ClusterRuleDecision StructureContractDetail::OrientationRuleFor::tryCalculateTransition(
    const StructureAnalysis& analysis, const AnalysisContext& context, int atomIndex, int neighborAtomIndex,
    int neighborIndex, Matrix3& outTransition) const {
    if (!handles(context.structureTypes->getInt(atomIndex))) return ClusterRuleDecision::Unhandled;
    return _rule->tryCalculateTransition(analysis, context, atomIndex, neighborAtomIndex, neighborIndex, outTransition);
}

StructureContractDetail::ContractRuleProvider::ContractRuleProvider(
    std::shared_ptr<const std::vector<OrientationClusterAtomState>> orientations, std::vector<int> orientationTypes,
    std::vector<int> cellTypes)
    : _orientations(orientations),
      _orientationRule(orientationTypes.empty() ? nullptr : std::make_shared<OrientationClusterRuleProvider>(std::move(orientations))),
      _orientationTypes(std::move(orientationTypes)),
      _cellTypes(std::move(cellTypes)),
      _minimumTrace(1.0 + 2.0 * std::cos(kCellMisorientationDegrees * PI / 180.0)) {}

void StructureContractDetail::ContractRuleProvider::initializeClusterSeed(
    const StructureAnalysis& analysis, const AnalysisContext& context, Cluster& cluster, int seedAtomIndex,
    int structureType) const {
    if (_orientationRule && isOrientation(structureType))
        _orientationRule->initializeClusterSeed(analysis, context, cluster, seedAtomIndex, structureType);
}

ClusterRuleDecision StructureContractDetail::ContractRuleProvider::tryAssignNeighbor(
    const StructureAnalysis& analysis, const AnalysisContext& context, const Cluster& cluster, int currentAtomIndex,
    int neighborAtomIndex, int neighborIndex, int structureType, int& outNeighborSymmetry) const {
    const int neighborType = context.structureTypes->getInt(neighborAtomIndex);
    if (isCell(structureType) || isCell(neighborType)) {
        if (!isCell(structureType) || !isCell(neighborType)) return ClusterRuleDecision::Rejected;
        return assignCellNeighbor(analysis, context, currentAtomIndex, neighborAtomIndex, neighborIndex, structureType,
                                  neighborType, outNeighborSymmetry);
    }
    // allowsCrossStructureGrowth() is global; other structure types never join across, as without a cell.
    if (neighborType != structureType) return ClusterRuleDecision::Rejected;
    if (!_orientationRule || !isOrientation(structureType)) return ClusterRuleDecision::Unhandled;
    return _orientationRule->tryAssignNeighbor(analysis, context, cluster, currentAtomIndex, neighborAtomIndex,
                                               neighborIndex, structureType, outNeighborSymmetry);
}

ClusterRuleDecision StructureContractDetail::ContractRuleProvider::tryCalculateTransition(
    const StructureAnalysis& analysis, const AnalysisContext& context, int atomIndex, int neighborAtomIndex,
    int neighborIndex, Matrix3& outTransition) const {
    const int type = context.structureTypes->getInt(atomIndex);
    const int neighborType = context.structureTypes->getInt(neighborAtomIndex);
    if (isCell(type) || isCell(neighborType)) {
        if (!isCell(type) || !isCell(neighborType)) return ClusterRuleDecision::Rejected;
        const Matrix3* atom = orientation(atomIndex);
        const Matrix3* neighbor = orientation(neighborAtomIndex);
        if (!atom || !neighbor || !orthonormalizeMatrix(Matrix3(neighbor->transposed() * *atom), outTransition))
            return ClusterRuleDecision::Rejected;
        return ClusterRuleDecision::Accepted;
    }
    if (!_orientationRule || !isOrientation(type)) return ClusterRuleDecision::Unhandled;
    return _orientationRule->tryCalculateTransition(analysis, context, atomIndex, neighborAtomIndex, neighborIndex, outTransition);
}

ClusterRuleDecision StructureContractDetail::ContractRuleProvider::assignCellNeighbor(
    const StructureAnalysis& analysis, const AnalysisContext& context, int current, int neighbor, int, int currentType,
    int neighborType, int& outNeighborSymmetry) const {
    const Matrix3* rc = orientation(current);
    const Matrix3* rn = orientation(neighbor);
    if (!rc || !rn) return ClusterRuleDecision::Rejected;
    Vector3 bond = context.simCell.wrapVector(context.positions->getPoint3(neighbor) - context.positions->getPoint3(current));
    const double length = bond.length();
    if (length <= EPSILON) return ClusterRuleDecision::Rejected;
    bond /= length;
    if (bondAlignment(analysis, currentType, *rc, bond) < kCellBondAlignment ||
        bondAlignment(analysis, neighborType, *rn, -bond) < kCellBondAlignment)
        return ClusterRuleDecision::Rejected;
    Matrix3 relative;
    if (!orthonormalizeMatrix(Matrix3(rn->transposed() * *rc), relative)) return ClusterRuleDecision::Rejected;
    if (relative(0, 0) + relative(1, 1) + relative(2, 2) <= _minimumTrace) return ClusterRuleDecision::Rejected;
    outNeighborSymmetry = 0;
    return ClusterRuleDecision::Accepted;
}

ContractSiteAlignment StructureContractDetail::alignSites(
    const LammpsParser::Frame& frame, const std::vector<int>& sites, const std::vector<int>& preferredTopology,
    const std::vector<LatticeTopology>& topologies, const NearestNeighborFinder& finder, int candidates,
    double rmsdCutoff, StructureContractAtoms& atoms, std::vector<OrientationClusterAtomState>& orientations,
    std::vector<std::array<int, MAX_NEIGHBORS>>& siteNeighbors) {
    ContractSiteAlignment alignment{topologies, rmsdCutoff};
    for (const int atom : sites) alignment.crystallineSites += preferredTopology[atom] != kNotCrystalline ? 1 : 0;

    tbb::enumerable_thread_specific<AlignmentTally> tallies([&] {
        AlignmentTally tally;
        tally.matched.assign(topologies.size(), 0);
        return tally;
    });
    ProgressReporter progress("Aligning crystalline sites", sites.size());
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, sites.size(), 256),
        [&](const tbb::blocked_range<std::size_t>& r) {
            AlignmentTally& tally = tallies.local();
            NearestNeighborFinder::Query<kNeighborQueryCapacity> query(finder);
            std::array<Vector3, kMaxMatchCandidates> deltas;
            std::array<int, kMaxMatchCandidates> deltaTypes;
            std::vector<LatticeMatch> matches(topologies.size());
            std::vector<unsigned char> aligned(topologies.size());
            for (std::size_t s = r.begin(); s < r.end(); ++s) {
                const int atom = sites[s];
                if (preferredTopology[atom] == kNotCrystalline) continue;
                query.findNeighbors(s);
                const auto& found = query.results();
                const int count = std::min(static_cast<int>(found.size()), candidates);
                for (int j = 0; j < count; ++j) {
                    deltas[j] = found[j].delta;
                    deltaTypes[j] = frame.types[sites[found[j].index]];
                }

                int bestTopology = -1;
                for (std::size_t t = 0; t < topologies.size(); ++t) {
                    aligned[t] = topologies[t].coordinationNumber <= count &&
                                 (topologies[t].siteType < 0 || topologies[t].siteType == frame.types[atom]) &&
                                 matchLattice(topologies[t], deltas.data(), matches[t], deltaTypes.data(), count);
                    if (aligned[t] && (bestTopology < 0 || matches[t].rmsd < matches[bestTopology].rmsd))
                        bestTopology = static_cast<int>(t);
                }
                if (bestTopology < 0) {
                    ++tally.unaligned;
                    continue;
                }
                // A class that names a multi-site cell prefers whichever of its sites fits best.
                int preferred = preferredTopology[atom];
                if (preferred >= 0 && topologies[preferred].cellSite >= 0) {
                    int bestSite = -1;
                    for (std::size_t t = 0; t < topologies.size(); ++t)
                        if (aligned[t] && topologies[t].cellSite >= 0 && topologies[t].name == topologies[preferred].name &&
                            (bestSite < 0 || matches[t].rmsd < matches[bestSite].rmsd))
                            bestSite = static_cast<int>(t);
                    if (bestSite >= 0) preferred = bestSite;
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
                orientations[atom].orientation = best.rotation;
                orientations[atom].valid = true;
                for (int k = 0; k < topologies[bestTopology].coordinationNumber; ++k)
                    siteNeighbors[s][k] = sites[found[best.slots[k]].index];
                if (overridden) tally.overrides.push_back(s);
            }
            progress.advance(r.size());
        });

    alignment.matched.assign(topologies.size(), 0);
    std::vector<std::size_t> overrides;
    for (const AlignmentTally& tally : tallies) {
        for (std::size_t t = 0; t < topologies.size(); ++t) alignment.matched[t] += tally.matched[t];
        alignment.rejected += tally.rejected;
        alignment.unaligned += tally.unaligned;
        alignment.rejectedRmsd.insert(alignment.rejectedRmsd.end(), tally.rejectedRmsd.begin(), tally.rejectedRmsd.end());
        overrides.insert(overrides.end(), tally.overrides.begin(), tally.overrides.end());
    }
    alignment.overrides = overrides.size();

    // Sites off their class's topology without the support of their shell become OTHER; the support
    // counts the assignments before any of them is dropped.
    const std::vector<int> assigned = atoms.topology;
    for (const std::size_t site : overrides) {
        const int atom = sites[site];
        const int taken = assigned[atom];
        int support = 0;
        for (int k = 0; k < topologies[taken].coordinationNumber; ++k)
            support += assigned[siteNeighbors[site][k]] == taken ? 1 : 0;
        if (support >= kOverrideSupport) continue;
        ++alignment.reverted;
        --alignment.matched[taken];
        atoms.topology[atom] = -1;
        orientations[atom].valid = false;
    }
    return alignment;
}

ContractGrainGrowth StructureContractDetail::growCellGrains(
    ContractSiteAlignment& alignment, const LammpsParser::Frame& frame, const std::vector<int>& sites,
    const std::vector<int>& siteIndex, const NearestNeighborFinder& finder, int candidates,
    const std::vector<LatticeTopology>& topologies, const std::vector<int>& preferredTopology, double rmsdCutoff,
    StructureContractAtoms& atoms, std::vector<OrientationClusterAtomState>& orientations,
    std::vector<std::array<int, MAX_NEIGHBORS>>& siteNeighbors) {
    std::vector<std::int64_t>& matched = alignment.matched;
    const double minimumTrace = 1.0 + 2.0 * std::cos(kCellMisorientationDegrees * PI / 180.0);
    std::map<std::pair<std::string, int>, int> topologyOfSite;
    for (std::size_t t = 0; t < topologies.size(); ++t)
        if (topologies[t].cellSite >= 0) topologyOfSite[{topologies[t].name, topologies[t].cellSite}] = static_cast<int>(t);
    auto cellOf = [&](int atom) -> const std::string* {
        const int preferred = preferredTopology[atom];
        return preferred >= 0 && topologies[preferred].cellSite >= 0 ? &topologies[preferred].name : nullptr;
    };
    std::vector<int> seeds;
    for (const int atom : sites)
        if (cellOf(atom) && atoms.topology[atom] >= 0 && topologies[atoms.topology[atom]].cellSite >= 0) seeds.push_back(atom);
    std::ranges::sort(seeds, [&](int a, int b) { return atoms.rmsd[a] < atoms.rmsd[b]; });

    NearestNeighborFinder::Query<kNeighborQueryCapacity> query(finder);
    std::array<Vector3, kMaxMatchCandidates> deltas;
    std::array<int, kMaxMatchCandidates> deltaTypes;
    std::vector<unsigned char> grown(static_cast<std::size_t>(frame.natoms), 0);
    std::vector<std::int64_t> grainOf(static_cast<std::size_t>(frame.natoms), 0);
    std::int64_t inconsistent = 0;
    std::deque<int> queue;
    std::int64_t regrown = 0;
    std::int64_t grains = 0;
    for (const int seed : seeds) {
        if (grown[seed]) continue;
        ++grains;
        grown[seed] = 1;
        grainOf[seed] = grains;
        queue.push_back(seed);
        while (!queue.empty()) {
            const int atom = queue.front();
            queue.pop_front();
            const LatticeTopology& own = topologies[atoms.topology[atom]];
            const Matrix3 guess = orientations[atom].orientation;
            for (int k = 0; k < own.coordinationNumber; ++k) {
                const int neighbor = siteNeighbors[siteIndex[atom]][k];
                const std::string* cell = cellOf(neighbor);
                if (grown[neighbor] || !cell || *cell != own.name) continue;
                query.findNeighbors(static_cast<std::size_t>(siteIndex[neighbor]));
                const auto& found = query.results();
                const int count = std::min(static_cast<int>(found.size()), candidates);
                for (int j = 0; j < count; ++j) {
                    deltas[j] = found[j].delta;
                    deltaTypes[j] = frame.types[sites[found[j].index]];
                }
                const auto target = topologyOfSite.find({own.name, own.neighborSites[k]});
                if (target == topologyOfSite.end()) continue;
                const int best = target->second;
                const LatticeTopology& site = topologies[best];
                LatticeMatch bestMatch;
                if (site.siteType != frame.types[neighbor] || site.coordinationNumber > count ||
                    !matchLatticeFrom(site, deltas.data(), guess, bestMatch, deltaTypes.data(), count) || bestMatch.rmsd > rmsdCutoff)
                    continue;
                const Matrix3 relative = guess.transposed() * bestMatch.rotation;
                if (relative(0, 0) + relative(1, 1) + relative(2, 2) <= minimumTrace) continue;
                bool consistent = true;
                for (int j = 0; j < site.coordinationNumber && consistent; ++j) {
                    const int other = sites[found[bestMatch.slots[j]].index];
                    if (grainOf[other] != grains) continue;
                    const LatticeTopology& otherSite = topologies[atoms.topology[other]];
                    consistent = otherSite.cellSite == site.neighborSites[j];
                    const auto& otherShell = siteNeighbors[siteIndex[other]];
                    for (int m = 0; m < otherSite.coordinationNumber && consistent; ++m)
                        if (otherShell[m] == neighbor)
                            consistent = (otherSite.latticeVectors[m] + site.latticeVectors[j]).length() < 1e-3;
                }
                if (!consistent) {
                    ++inconsistent;
                    continue;
                }
                const int previous = atoms.topology[neighbor];
                if (previous != best || !orientations[neighbor].valid ||
                    !Matrix3(orientations[neighbor].orientation.transposed() * bestMatch.rotation).equals(Matrix3(Matrix3::Identity{}), 0.05))
                    ++regrown;
                if (previous >= 0) --matched[previous];
                ++matched[best];
                atoms.topology[neighbor] = best;
                atoms.rmsd[neighbor] = static_cast<float>(bestMatch.rmsd);
                orientations[neighbor].orientation = bestMatch.rotation;
                orientations[neighbor].valid = true;
                for (int j = 0; j < topologies[best].coordinationNumber; ++j)
                    siteNeighbors[siteIndex[neighbor]][j] = sites[found[bestMatch.slots[j]].index];
                grown[neighbor] = 1;
                grainOf[neighbor] = grains;
                queue.push_back(neighbor);
            }
        }
    }
    // Growth crosses any boundary within the cluster misorientation, so two grains of one cell that meet
    // within it are one crystal whose registrations disagree (a pseudo-translation apart, or a seed's own
    // ambiguous fit): circuits through both would not close on a lattice vector. The smaller one is no
    // lattice site, nor is a seed no neighbour joined. Grains are dropped smallest first.
    std::vector<std::int64_t> grainSize(static_cast<std::size_t>(grains) + 1, 0);
    for (const int atom : sites) grainSize[static_cast<std::size_t>(grainOf[atom])]++;
    std::vector<std::vector<int>> members(static_cast<std::size_t>(grains) + 1);
    for (const int atom : sites)
        if (grainOf[atom] > 0) members[static_cast<std::size_t>(grainOf[atom])].push_back(atom);
    std::vector<std::int64_t> order;
    for (std::int64_t g = 1; g <= grains; ++g) order.push_back(g);
    std::ranges::sort(order, [&](std::int64_t a, std::int64_t b) { return grainSize[a] < grainSize[b]; });
    std::vector<unsigned char> dropped(static_cast<std::size_t>(grains) + 1, 0);
    std::int64_t droppedGrains = 0;
    for (const std::int64_t g : order) {
        bool conflict = grainSize[g] == 1;
        for (std::size_t i = 0; i < members[g].size() && !conflict; ++i) {
            const int atom = members[g][i];
            const LatticeTopology& own = topologies[atoms.topology[atom]];
            for (int k = 0; k < own.coordinationNumber && !conflict; ++k) {
                const int other = siteNeighbors[siteIndex[atom]][k];
                const std::int64_t h = grainOf[other];
                if (h == 0 || h == g || dropped[h] || grainSize[h] < grainSize[g]) continue;
                const Matrix3 relative = orientations[atom].orientation.transposed() * orientations[other].orientation;
                conflict = relative(0, 0) + relative(1, 1) + relative(2, 2) > minimumTrace;
            }
        }
        if (!conflict) continue;
        dropped[g] = 1;
        ++droppedGrains;
        for (const int atom : members[g]) {
            --matched[atoms.topology[atom]];
            atoms.topology[atom] = -1;
            orientations[atom].valid = false;
            grainOf[atom] = 0;
        }
    }
    return {alignment, regrown, grains - droppedGrains, inconsistent};
}

std::optional<std::string> writeStructureContract(
    const LammpsParser::Frame& frame,
    const std::string& outputBase,
    const std::vector<int>& sites,
    const std::vector<int>& preferredTopology,
    const std::vector<LatticeTopology>& topologies,
    double rmsdCutoff,
    json& result,
    StructureContractAtoms& atoms,
    CnatLog& log) {
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
    // Sites of a cell of several species align onto the nearest of each type among more candidates.
    const bool typedSites = std::ranges::any_of(topologies, [](const LatticeTopology& t) { return !t.neighborTypes.empty(); });
    const int candidates = typedSites ? std::min(kMaxMatchCandidates, maxCn + kExtraCandidates) : maxCn;
    // Weighed by the sample runs.
    ProgressSteps steps({
        {"Indexing the lattice sites", 1.0},
        {"Aligning crystalline sites", 12.0},
        {"Building clusters", 3.0},
        {"Writing the structure contract", 18.0},
    });
    NearestNeighborFinder finder(candidates);
    if (!finder.prepare(sitePositions.get(), frame.simulationCell))
        return std::string("NearestNeighborFinder::prepare failed on the lattice sites");
    steps.next();

    atoms.topology.assign(natoms, -1);
    atoms.rmsd.assign(natoms, std::numeric_limits<float>::quiet_NaN());
    std::vector<std::array<int, MAX_NEIGHBORS>> siteNeighbors(sites.size());
    auto orientations = std::make_shared<std::vector<OrientationClusterAtomState>>(natoms);
    ContractSiteAlignment alignment = alignSites(frame, sites, preferredTopology, topologies, finder, candidates, rmsdCutoff,
                                                 atoms, *orientations, siteNeighbors);
    log.dump(alignment);

    std::vector<int> siteIndex(natoms, -1);
    for (std::size_t s = 0; s < sites.size(); ++s) siteIndex[sites[s]] = static_cast<int>(s);
    const bool cellSites = std::ranges::any_of(topologies, [](const LatticeTopology& t) { return t.cellSite >= 0; });
    const ContractGrainGrowth growth = cellSites
        ? growCellGrains(alignment, frame, sites, siteIndex, finder, candidates, topologies, preferredTopology, rmsdCutoff,
                         atoms, *orientations, siteNeighbors)
        : ContractGrainGrowth{alignment};
    log.dump(growth);

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

    steps.next();
    ClusterInputAdapterUtils::prepareSymmetryAwareClusterInputs(analysis, context, true,
        [&](std::size_t atomIndex, int structureType) {
            return structureType != LATTICE_OTHER && analysis.numberOfNeighbors(static_cast<int>(atomIndex)) > 0;
        });
    // Topologies whose overlap-grown bonds do not span space (sc, st, a7) join their atoms by the shared
    // orientation rule instead, as PTM does for SC.
    std::vector<int> orientationTypes;
    std::vector<int> cellTypes;
    for (const LatticeTopology& topology : topologies) {
        if (topology.cellSite >= 0) {
            cellTypes.push_back(topology.structureType);
            continue;
        }
        if (!topology.orientationClusters) continue;
        orientationTypes.push_back(topology.structureType);
    }
    log.dump(ContractClusterRules{topologies});
    if (!cellTypes.empty()) {
        analysis.setClusterRuleProvider(std::make_shared<ContractRuleProvider>(orientations, std::move(orientationTypes), std::move(cellTypes)));
    } else if (!orientationTypes.empty()) {
        analysis.setClusterRuleProvider(std::make_shared<OrientationRuleFor>(orientations, std::move(orientationTypes)));
    }
    ClusterBuilder clusterBuilder(analysis, context);
    BuiltClusters built = clusterBuilder.build(false);
    normalizeReconstructedClusterGraphForExport(analysis, context);
    built.latticeSites = &sites;
    log.dump(built);

    // Atoms off the neighbour sublattice are no lattice sites of the topologies; left in the contract as
    // OTHER they would turn every Delaunay cell into a defect cell.
    std::vector<bool> contractSites;
    const std::vector<bool>* contractMask = nullptr;
    if (sites.size() < natoms) {
        contractSites.assign(natoms, false);
        for (const int atom : sites) contractSites[atom] = true;
        contractMask = &contractSites;
    }

    steps.next();
    const auto outputs = AnalysisPipelineUtils::appendClusterOutputs(frame, outputBase, outputBase + "_annotated.dump",
        context, analysis, result, &error, {}, contractMask);
    if (!outputs)
        return error;
    log.dump(outputs);

    const auto& clusters = analysis.clusterGraph().clusters();
    // The cluster graph always holds the null cluster 0.
    const std::size_t clusterCount = clusters.empty() ? 0 : clusters.size() - 1;
    const std::int64_t aligned = std::accumulate(alignment.matched.begin(), alignment.matched.end(), std::int64_t{0});

    json& listing = result["main_listing"];
    listing["lattice_topologies"] = names;
    listing["lattice_rmsd_cutoff"] = rmsdCutoff;
    listing["lattice_sites"] = static_cast<std::int64_t>(sites.size());
    listing["aligned_sites"] = aligned;
    listing["sites_over_rmsd_cutoff"] = alignment.rejected;
    listing["sites_off_their_class_topology"] = static_cast<std::int64_t>(alignment.overrides) - alignment.reverted;
    listing["clusters"] = static_cast<std::int64_t>(clusterCount);
    json rows = json::array();
    for (std::size_t t = 0; t < topologies.size(); ++t) {
        rows.push_back({
            {"topology", topologies[t].name},
            {"structure_id", topologies[t].structureType},
            {"coordination_number", topologies[t].coordinationNumber},
            {"count", alignment.matched[t]},
            {"fraction", alignment.crystallineSites
                ? static_cast<double>(alignment.matched[t]) / static_cast<double>(alignment.crystallineSites) : 0.0},
        });
        if (topologies[t].cellSite >= 0) rows.back()["cell_site"] = topologies[t].cellSite;
    }
    result["sub_listings"]["lattice_alignment"] = std::move(rows);
    return std::nullopt;
}

}
