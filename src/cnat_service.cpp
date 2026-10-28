#include <volt/plugin/cnat_service.h>
#include <volt/plugin/cnat_model.h>
#include <volt/plugin/phase_field_mesh.h>
#include <volt/plugin/lattice_matcher.h>
#include <volt/plugin/structure_contract.h>
#include <volt/analysis/nearest_neighbor_finder.h>
#include <volt/core/analysis_result.h>
#include <volt/core/frame_adapter.h>
#include <volt/core/progress.h>
#include <volt/plugin/output_serializer.h>
#include <volt/utilities/json_utils.h>
#include <volt/utilities/parquet_atom_writer.h>
#include <volt/utilities/parquet_mesh_writer.h>
#include <volt/structures/crystal_topology_registry.h>
#include <spdlog/spdlog.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <optional>
#include <utility>

namespace Volt {

namespace {

constexpr int kMaxNeighbors = 128;
constexpr const char* kParquetSourceKey = "__parquet_source__";

std::vector<std::string> classNames(const CnatModel& model) {
    std::vector<std::string> names;
    const json meta = json::parse(model.metadata(), nullptr, false);
    if (meta.is_object() && meta.contains("classes") && meta["classes"].is_array())
        for (const auto& c : meta["classes"])
            if (c.is_string()) names.push_back(c.get<std::string>());
    while (names.size() < static_cast<std::size_t>(model.config().classes))
        names.push_back("CLASS_" + std::to_string(names.size()));
    names.resize(model.config().classes);
    return names;
}

std::string modelName(const CnatModel& model) {
    const json meta = json::parse(model.metadata(), nullptr, false);
    if (meta.is_object() && meta.contains("name") && meta["name"].is_string())
        return meta["name"].get<std::string>();
    return "unnamed";
}

// Comma-separated atom types; empty selects every atom.
std::optional<std::vector<int>> parseTypes(const std::string& text) {
    std::vector<int> types;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t end = std::min(text.find(',', start), text.size());
        const std::string item = text.substr(start, end - start);
        if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos)
            return std::nullopt;
        types.push_back(std::stoi(item));
        start = end + 1;
    }
    return types;
}

std::string lowercase(std::string s) {
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// Structure id of every class: StructureType's when the class names one of its structures (FCC, HCP,
// BCC, ...), so CNAT's atoms read as the other producers'; class 0 is OTHER. The other crystal classes
// of the built-in models keep one id each across models (plugin.json names them); those of any other
// model count up from kCustomTopologyTypeBase + 100.
std::vector<int> structureIds(const std::vector<std::string>& names) {
    static const std::array<const char*, 4> kNamedPhases = {"OLIVINE", "KYANITE", "ANDALUSITE", "SILLIMANITE"};
    std::vector<int> ids(names.size(), StructureType::OTHER);
    for (std::size_t c = 1; c < names.size(); ++c) {
        const int known = structureTypeOfName(names[c]);
        const auto named = std::ranges::find(kNamedPhases, names[c]);
        ids[c] = known != StructureType::OTHER ? known
               : named != kNamedPhases.end() ? kCustomTopologyTypeBase + 1 + static_cast<int>(named - kNamedPhases.begin())
               : kCustomTopologyTypeBase + 100 + static_cast<int>(c);
    }
    return ids;
}

std::vector<std::string> splitList(const std::string& text) {
    std::vector<std::string> items;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = std::min(text.find(',', start), text.size());
        std::string item = text.substr(start, end - start);
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) items.push_back(std::move(item));
        start = end + 1;
    }
    return items;
}

// The crystal a crystal class of the model was trained on (export_model.py --crystal): a lattice topology
// of the registry, or the reference cell its training crystal was built from.
struct ClassCrystal {
    std::string topology;
    std::optional<CellAtoms> cell;
};

std::vector<ClassCrystal> modelCrystals(const CnatModel& model, const std::vector<std::string>& names) {
    std::vector<ClassCrystal> crystals(names.size());
    const json meta = json::parse(model.metadata(), nullptr, false);
    if (!meta.is_object() || !meta.contains("crystals") || !meta["crystals"].is_object()) return crystals;
    for (std::size_t c = 1; c < names.size(); ++c) {
        const auto entry = meta["crystals"].find(names[c]);
        if (entry == meta["crystals"].end() || !entry->is_object()) continue;
        if (entry->contains("topology")) crystals[c].topology = lowercase((*entry)["topology"].get<std::string>());
        if (!entry->contains("cell") || !entry->contains("atoms")) continue;
        CellAtoms cell;
        const auto& rows = (*entry)["cell"];
        cell.cell = Matrix3(Vector3(rows[0][0], rows[0][1], rows[0][2]), Vector3(rows[1][0], rows[1][1], rows[1][2]),
                            Vector3(rows[2][0], rows[2][1], rows[2][2]));
        for (const auto& atom : (*entry)["atoms"]) {
            cell.types.push_back(atom[0].get<int>());
            cell.positions.emplace_back(atom[1].get<double>(), atom[2].get<double>(), atom[3].get<double>());
        }
        crystals[c].cell = std::move(cell);
    }
    return crystals;
}

// The sublattice of a cell the structure contract is written on: the species with the fewest atoms in
// the cell. It is the sparsest one, whose sites are farthest apart and least alike under the other
// sublattices' pseudo-symmetries (forsterite's oxygens are nearly hexagonal close packed, its Si are
// not), and one Wyckoff orbit has every translation of the crystal, so its Burgers vectors are the
// crystal's.
int contractSpecies(const CellAtoms& cell) {
    std::map<int, int> counts;
    for (const int type : cell.types) counts[type]++;
    int species = -1;
    for (const auto& [type, count] : counts)
        if (species < 0 || count < counts[species]) species = type;
    return species;
}

// The lattice topologies of the model's crystal classes, each close-packed or diamond one with its
// partner: the planar faults of either are local patches of the other, as CNA labels both from
// either input lattice.
std::vector<std::string> modelTopologies(const std::vector<std::string>& names, const std::vector<ClassCrystal>& crystals) {
    static const std::pair<const char*, const char*> partners[] = {
        {"fcc", "hcp"}, {"hcp", "fcc"}, {"cubic_diamond", "hex_diamond"}, {"hex_diamond", "cubic_diamond"}};
    std::vector<std::string> topologies;
    auto add = [&](const std::string& name) {
        if (crystalTopologyByName(name) && std::ranges::find(topologies, name) == topologies.end())
            topologies.push_back(name);
    };
    const bool declared = std::ranges::any_of(crystals, [](const ClassCrystal& c) { return !c.topology.empty() || c.cell; });
    for (std::size_t c = 1; c < names.size(); ++c) {
        // A model that stores its crystals names each class's topology; an older one, by the class name.
        const std::string name = declared ? crystals[c].topology : lowercase(names[c]);
        if (name.empty() || !crystalTopologyByName(name)) continue;
        add(name);
        for (const auto& [topology, partner] : partners)
            if (name == topology) add(partner);
    }
    return topologies;
}

// probability_crystalline (1 - P(class 0), the field of the interface mesh, with the same name for
// every model) and one probability_<class> column per class.
void writeProbabilities(ColumnarAtomWriter& w, const float* p, const std::vector<std::string>& names) {
    w.field("probability_crystalline", 1.0 - static_cast<double>(p[0]));
    for (std::size_t c = 0; c < names.size(); ++c)
        w.field("probability_" + lowercase(names[c]), static_cast<double>(p[c]));
}

// Class probabilities of every point, row-major [point][class].
template<typename PointAt>
std::vector<float> classifyPoints(const CnatModel& model, const NearestNeighborFinder& finder,
                                  std::size_t count, PointAt pointAt, const char* label) {
    const int k = model.config().neighbors;
    const int classes = model.config().classes;
    std::vector<float> probabilities(count * classes);
    ProgressReporter progress(label, count);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count, 64),
        [&](const tbb::blocked_range<std::size_t>& r) {
            CnatModel::Workspace ws;
            NearestNeighborFinder::Query<kMaxNeighbors> query(finder);
            std::vector<float> deltas(3 * static_cast<std::size_t>(k));
            for (std::size_t i = r.begin(); i < r.end(); ++i) {
                query.findNeighbors(pointAt(i), true);
                const auto& neighbors = query.results();
                for (int j = 0; j < k; ++j) {
                    deltas[3 * j + 0] = static_cast<float>(neighbors[j].delta.x());
                    deltas[3 * j + 1] = static_cast<float>(neighbors[j].delta.y());
                    deltas[3 * j + 2] = static_cast<float>(neighbors[j].delta.z());
                }
                model.classify(deltas.data(), ws, probabilities.data() + i * classes);
            }
            progress.advance(r.size());
        });
    return probabilities;
}

// Class 0 is the disordered phase. A point is crystalline when 1 - P(class 0) reaches the threshold,
// and then takes its most probable crystal class; for a binary model this is P(class 1) >= threshold.
std::vector<int> assignLabels(const std::vector<float>& probabilities, int classes, double threshold) {
    const std::size_t count = probabilities.size() / classes;
    std::vector<int> labels(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float* p = probabilities.data() + i * classes;
        labels[i] = 1.0 - p[0] >= threshold
            ? static_cast<int>(std::max_element(p + 1, p + classes) - p)
            : 0;
    }
    return labels;
}

std::string countSummary(const std::vector<std::int64_t>& counts, const std::vector<std::string>& names, std::size_t total) {
    std::string line;
    for (std::size_t c = 0; c < counts.size(); ++c) {
        char percent[32];
        std::snprintf(percent, sizeof(percent), " (%.1f%%)", percentOf(static_cast<std::uint64_t>(counts[c]), total));
        if (!line.empty()) line += ", ";
        line += names[c] + " " + std::to_string(counts[c]) + percent;
    }
    return line;
}

json countRows(const std::vector<std::int64_t>& counts, const std::vector<std::string>& names,
               const std::vector<int>& ids, std::size_t total, const char* countKey) {
    json rows = json::array();
    for (std::size_t c = 0; c < counts.size(); ++c) {
        rows.push_back({
            {"structure_id", ids[c]},
            {"structure_name", names[c]},
            {countKey, counts[c]},
            {"fraction", total ? static_cast<double>(counts[c]) / static_cast<double>(total) : 0.0},
        });
    }
    return rows;
}

json cellJson(const SimulationCell& cell) {
    const auto& m = cell.matrix();
    const auto& pbc = cell.pbcFlags();
    return {
        {"matrix", {
            {m.column(0).x(), m.column(0).y(), m.column(0).z()},
            {m.column(1).x(), m.column(1).y(), m.column(1).z()},
            {m.column(2).x(), m.column(2).y(), m.column(2).z()},
        }},
        {"origin", {m.column(3).x(), m.column(3).y(), m.column(3).z()}},
        {"pbc", {pbc[0], pbc[1], pbc[2]}},
    };
}

void writeMesh(const std::string& filePath, const PhaseFieldMesh& mesh, const SimulationCell& cell) {
    const std::string base = filePath.substr(0, filePath.size() - std::string(".parquet").size());
    const std::string verticesPath = base + ".vertices.parquet";
    const std::string facetsPath = base + ".facets.parquet";
    streamMeshTablesToParquet(verticesPath, facetsPath, mesh.vertices, mesh.facets);

    json doc;
    doc["main_listing"] = {
        {"total_nodes", static_cast<std::int64_t>(mesh.vertices.size())},
        {"total_facets", static_cast<std::int64_t>(mesh.facets.size())},
        {"area", mesh.area},
    };
    doc["export"]["MeshExporter"][kParquetSourceKey] = {
        {"vertices", std::filesystem::path(verticesPath).filename().string()},
        {"facets", std::filesystem::path(facetsPath).filename().string()},
        {"cell", cellJson(cell)},
    };
    JsonUtils::writeJsonToParquet(doc, filePath);
}

}

json CnatService::compute(const LammpsParser::Frame& frame, const std::string& outputBase) {
    if (frame.natoms <= 0)
        return AnalysisResult::failure("Invalid number of atoms");
    if (!(_threshold > 0.0 && _threshold < 1.0))
        return AnalysisResult::failure("--threshold must lie strictly between 0 and 1");
    if (!(_rmsd > 0.0))
        return AnalysisResult::failure("--rmsd must be positive");

    std::optional<CnatModel> loaded;
    try {
        loaded = CnatModel::load(_modelPath);
    } catch (const std::exception& e) {
        return AnalysisResult::failure(std::string("Cannot load the CNAT model: ") + e.what());
    }
    const CnatModel& model = *loaded;
    const CnatConfig& cfg = model.config();
    const std::vector<std::string> names = classNames(model);
    const std::vector<int> ids = structureIds(names);
    const bool builtIn = std::ranges::any_of(CnatModel::embeddedModels(),
        [&](const CnatModel::Embedded& e) { return _modelPath.empty() || _modelPath == e.name; });
    spdlog::info("Model {}: {} neighbours, hidden {}, {} layers, {} heads, MLP {}, {} parameters, classes {}",
        builtIn ? modelName(model) + " (built in)" : _modelPath,
        cfg.neighbors, cfg.hidden, cfg.layers, cfg.heads, cfg.mlp, model.parameterCount(), json(names).dump());

    // The neighbour types default to those the model was trained with (export_model.py --neighbor_types).
    std::string typeList = _types;
    if (typeList.empty()) {
        const json meta = json::parse(model.metadata(), nullptr, false);
        if (meta.is_object() && meta.contains("neighbor_types") && meta["neighbor_types"].is_array())
            for (const auto& t : meta["neighbor_types"]) typeList += (typeList.empty() ? "" : ",") + std::to_string(t.get<int>());
        if (!typeList.empty()) spdlog::info("Neighbour types {}, those the model was trained with", typeList);
    }
    const auto types = parseTypes(typeList);
    if (!types)
        return AnalysisResult::failure("--types must be a comma-separated list of atom types, got '" + _types + "'");
    LammpsParser::Frame sublattice;
    const LammpsParser::Frame* cloud = &frame;
    if (!types->empty()) {
        if (frame.types.size() < static_cast<std::size_t>(frame.natoms))
            return AnalysisResult::failure("--types needs a type column in the dump");
        sublattice.simulationCell = frame.simulationCell;
        for (int i = 0; i < frame.natoms; ++i)
            if (std::ranges::find(*types, frame.types[i]) != types->end())
                sublattice.positions.push_back(frame.positions[i]);
        sublattice.natoms = static_cast<int>(sublattice.positions.size());
        spdlog::info("Neighbours are the {} atoms of type {} ({:.1f}% of the frame)", sublattice.natoms, typeList,
            percentOf(static_cast<std::uint64_t>(sublattice.natoms), static_cast<std::size_t>(frame.natoms)));
        cloud = &sublattice;
    }
    if (cloud->natoms < cfg.neighbors)
        return AnalysisResult::failure("The neighbour atoms are fewer than the model's " + std::to_string(cfg.neighbors));

    auto positions = FrameAdapter::createPositionPropertyShared(*cloud);
    if (!positions)
        return AnalysisResult::failure("Failed to create position property");

    spdlog::info("Building nearest-neighbour index ({} neighbours) for {} atoms...", cfg.neighbors, cloud->natoms);
    Stopwatch timer;
    NearestNeighborFinder finder(cfg.neighbors);
    if (!finder.prepare(positions.get(), frame.simulationCell))
        return AnalysisResult::failure("NearestNeighborFinder::prepare failed");
    spdlog::info("Nearest-neighbour index ready in {}", timer.elapsed());

    const std::size_t natoms = static_cast<std::size_t>(frame.natoms);
    spdlog::info("Classifying {} atoms...", natoms);
    timer.restart();
    const std::vector<float> atomProbabilities = classifyPoints(model, finder, natoms,
        [&](std::size_t i) { return frame.positions[i]; }, "Classifying atoms");
    const std::vector<int> atomLabels = assignLabels(atomProbabilities, cfg.classes, _threshold);
    spdlog::info("Classified {} atoms in {} ({:.1f} us per atom)", natoms, timer.elapsed(),
        1e6 * std::chrono::duration<double>(timer.elapsedTime()).count() / static_cast<double>(natoms));

    std::vector<std::int64_t> atomCounts(cfg.classes, 0);
    for (int label : atomLabels) atomCounts[label]++;
    spdlog::info("Atoms: {}", countSummary(atomCounts, names, natoms));

    json result = AnalysisResult::success();
    result["main_listing"] = {
        {"total_atoms", frame.natoms},
        {"model", builtIn ? modelName(model) : _modelPath},
        {"neighbors", cfg.neighbors},
        {"threshold", _threshold},
        {"neighbour_types", typeList.empty() ? std::string("all") : typeList},
    };
    for (int c = 0; c < cfg.classes; ++c)
        result["main_listing"][names[c]] = atomCounts[c];
    result["sub_listings"]["structure_counts"] = countRows(atomCounts, names, ids, natoms, "count");

    std::vector<LatticeTopology> topologies;
    StructureContractAtoms contractAtoms;
    const std::vector<ClassCrystal> crystals = modelCrystals(model, names);
    std::vector<int> cellSpecies;
    const std::string topologyRequest = lowercase(_topologies);
    if (!outputBase.empty() && topologyRequest != "none") {
        const bool automatic = topologyRequest.empty() || topologyRequest == "auto";
        std::vector<std::string> requested;
        try {
            if (!_latticeDir.empty()) setCrystalTopologySearchRoot(_latticeDir);
            requested = automatic ? modelTopologies(names, crystals) : splitList(_topologies);
            topologies = loadLatticeTopologies(requested);
            if (requested.empty() && _cell.empty() && std::ranges::none_of(crystals, [](const ClassCrystal& c) { return c.cell.has_value(); }))
                spdlog::info("No class of the model names a lattice topology; pass --topologies to write the structure contract");
        } catch (const std::exception& e) {
            if (!automatic)
                return AnalysisResult::failure(std::string("Cannot load the lattice topologies: ") + e.what());
            spdlog::warn("Cannot load the lattice topologies ({}); the structure contract is not written", e.what());
        }
        for (std::size_t c = 1; c < names.size() && automatic; ++c) {
            if (!crystals[c].cell) continue;
            const int species = contractSpecies(*crystals[c].cell);
            try {
                CellTopologies cell = cellTopologies(*crystals[c].cell, lowercase(names[c]), {species}, 0.0,
                    kCustomTopologyTypeBase + static_cast<int>(topologies.size()));
                int least = MAX_NEIGHBORS, most = 0;
                for (const LatticeTopology& site : cell.sites) {
                    least = std::min(least, site.coordinationNumber);
                    most = std::max(most, site.coordinationNumber);
                }
                spdlog::info("Crystal of {} (stored with the model): its {} sites of type {}, the sparsest sublattice, "
                             "neighbours within {:.3f} A ({} to {} per site)", names[c], cell.sites.size(), species,
                             cell.cutoff, least, most);
                if (std::ranges::find(cellSpecies, species) == cellSpecies.end()) cellSpecies.push_back(species);
                for (LatticeTopology& site : cell.sites) topologies.push_back(std::move(site));
            } catch (const std::exception& e) {
                return AnalysisResult::failure("Cannot use the crystal of " + names[c] + ": " + e.what());
            }
        }
        if (!_cell.empty()) {
            const auto cellTypes = parseTypes(_cellTypes);
            if (!cellTypes)
                return AnalysisResult::failure("--cell_types must be a comma-separated list of atom types, got '" + _cellTypes + "'");
            try {
                CellTopologies cell = loadCellTopologies(_cell, *cellTypes, _cellCutoff,
                    kCustomTopologyTypeBase + static_cast<int>(topologies.size()));
                int least = MAX_NEIGHBORS, most = 0;
                for (const LatticeTopology& site : cell.sites) {
                    least = std::min(least, site.coordinationNumber);
                    most = std::max(most, site.coordinationNumber);
                }
                spdlog::info("Reference cell {}: {} sites of type {}, neighbours within {:.3f} A ({} to {} per site)", cell.name,
                             cell.sites.size(), _cellTypes.empty() ? std::string("any") : _cellTypes, cell.cutoff, least, most);
                result["main_listing"]["cell"] = cell.name;
                result["main_listing"]["cell_cutoff"] = cell.cutoff;
                for (LatticeTopology& site : cell.sites) topologies.push_back(std::move(site));
            } catch (const std::exception& e) {
                return AnalysisResult::failure(std::string("Cannot load the reference cell: ") + e.what());
            }
        }
        if (!topologies.empty()) {
            const auto contractTypes = parseTypes(_contractTypes);
            if (!contractTypes)
                return AnalysisResult::failure("--contract_types must be a comma-separated list of atom types, got '" + _contractTypes + "'");
            // The contract sublattice is the given one, else that of the model's cells with the model's
            // neighbour sublattice for the classes with a registry topology, else the latter.
            const bool registryClasses = std::ranges::any_of(topologies, [](const LatticeTopology& t) { return t.cellSite < 0; });
            std::vector<int> siteTypes = *contractTypes;
            if (siteTypes.empty() && !cellSpecies.empty() && !(registryClasses && types->empty())) {
                siteTypes = cellSpecies;
                if (registryClasses) siteTypes.insert(siteTypes.end(), types->begin(), types->end());
            } else if (siteTypes.empty()) {
                siteTypes = *types;
            }
            std::vector<int> sites;
            for (int i = 0; i < frame.natoms; ++i)
                if (siteTypes.empty() || std::ranges::find(siteTypes, frame.types[i]) != siteTypes.end()) sites.push_back(i);
            std::vector<int> classTopology(names.size(), kAnyTopology);
            classTopology[0] = kNotCrystalline;
            for (std::size_t c = 1; c < names.size(); ++c)
                for (std::size_t t = 0; t < topologies.size(); ++t)
                    if (classTopology[c] < 0 && lowercase(names[c]) == lowercase(topologies[t].name)) classTopology[c] = static_cast<int>(t);
            // With a cell, the classes it is not (andalusite next to kyanite) are no lattice sites of it.
            if (!_cell.empty() || !cellSpecies.empty())
                for (std::size_t c = 1; c < names.size(); ++c)
                    if (classTopology[c] == kAnyTopology) classTopology[c] = kNotCrystalline;
            std::vector<int> preferred(natoms);
            for (std::size_t i = 0; i < natoms; ++i) preferred[i] = classTopology[atomLabels[i]];
            if (auto error = writeStructureContract(frame, outputBase, sites, preferred, topologies, _rmsd, result, contractAtoms))
                return AnalysisResult::failure("Cannot write the structure contract: " + *error);
        }
    }
    result["main_listing"]["structure_contract"] = !topologies.empty();

    if (_gridSpacing > 0.0) {
        const SimulationCell& cell = frame.simulationCell;
        std::array<int, 3> dims{};
        for (int d = 0; d < 3; ++d)
            dims[d] = std::max(2, static_cast<int>(std::lround(cell.matrix().column(d).length() / _gridSpacing)));
        const std::size_t gridCount = static_cast<std::size_t>(dims[0]) * dims[1] * dims[2];
        auto gridPoint = [&](std::size_t g) {
            const std::size_t i = g % dims[0];
            const std::size_t j = (g / dims[0]) % dims[1];
            const std::size_t k = g / (static_cast<std::size_t>(dims[0]) * dims[1]);
            return cell.reducedToAbsolute(gridPointReduced(dims, static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)));
        };

        spdlog::info("Evaluating the phase field on a {}x{}x{} grid ({} points, about {:.2f} A apart)...",
            dims[0], dims[1], dims[2], gridCount, _gridSpacing);
        timer.restart();
        const std::vector<float> gridProbabilities = classifyPoints(model, finder, gridCount, gridPoint, "Evaluating the phase field");
        const std::vector<int> gridLabels = assignLabels(gridProbabilities, cfg.classes, _threshold);
        std::vector<std::int64_t> gridCounts(cfg.classes, 0);
        for (int label : gridLabels) gridCounts[label]++;
        spdlog::info("Phase field ready in {}; volume: {}", timer.elapsed(), countSummary(gridCounts, names, gridCount));

        result["main_listing"]["grid"] = std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + "x" + std::to_string(dims[2]);
        result["main_listing"]["grid_spacing"] = _gridSpacing;
        result["sub_listings"]["volume_fractions"] = countRows(gridCounts, names, ids, gridCount, "grid_points");

        if (_exportInterfaceMesh) {
            spdlog::info("Extracting the interface at 1 - P({}) = {} (smoothing {:.1f} A)...", names[0], _threshold, _interfaceSmoothing);
            timer.restart();
            std::vector<float> field(gridCount);
            for (std::size_t g = 0; g < gridCount; ++g) field[g] = 1.0f - gridProbabilities[g * cfg.classes];
            if (_interfaceSmoothing > 0.0)
                field = smoothField(field, dims, cell, _interfaceSmoothing);
            const PhaseFieldMesh mesh = extractIsosurface(field, dims, cell, static_cast<float>(_threshold));
            spdlog::info("  {} vertices, {} facets, area {:.1f} A^2 in {}",
                mesh.vertices.size(), mesh.facets.size(), mesh.area, timer.elapsed());
            result["main_listing"]["interface_area"] = mesh.area;
            if (!outputBase.empty())
                writeMesh(outputBase + "_interface_mesh.parquet", mesh, cell);
        }

        if (_exportPhaseField && !outputBase.empty()) {
            LammpsParser::Frame grid;
            grid.timestep = frame.timestep;
            grid.natoms = static_cast<int>(gridCount);
            grid.simulationCell = cell;
            grid.positions.resize(gridCount);
            grid.ids.resize(gridCount);
            grid.types.assign(gridCount, 1);
            for (std::size_t g = 0; g < gridCount; ++g) {
                grid.positions[g] = gridPoint(g);
                grid.ids[g] = static_cast<int>(g + 1);
            }
            spdlog::info("Writing the phase field ({} points)", gridCount);
            streamAtomsToParquet(outputBase + "_phase_field.parquet", grid,
                [&](std::size_t g) { return names[gridLabels[g]]; },
                [&](ColumnarAtomWriter& w, std::size_t g) {
                    writeProbabilities(w, gridProbabilities.data() + g * cfg.classes, names);
                },
                [&](std::size_t g) { return ids[gridLabels[g]]; },
                true);
        }
    }

    if (!outputBase.empty()) {
        Plugin::serializePluginOutput(outputBase, frame, result, {
            .summaryFileSuffix = "_cnat",
            .bucketResolver = [&](std::size_t i) { return names[atomLabels[i]]; },
            .perAtomColumnWriter = [&](ColumnarAtomWriter& w, std::size_t i) {
                writeProbabilities(w, atomProbabilities.data() + i * cfg.classes, names);
                if (topologies.empty()) return;
                const int t = contractAtoms.topology[i];
                w.field("lattice_structure", t >= 0 ? topologies[t].name : std::string("OTHER"));
                if (!std::isnan(contractAtoms.rmsd[i]))
                    w.field("lattice_rmsd", static_cast<double>(contractAtoms.rmsd[i]));
            },
            .resolveStructureId = [&](std::size_t i) { return ids[atomLabels[i]]; },
            .includeStructureColumns = true,
        });
    }
    return result;
}

}
