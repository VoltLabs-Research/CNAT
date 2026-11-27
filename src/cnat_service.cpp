#include <volt/plugin/cnat_service.h>
#include <volt/plugin/cnat_log.h>
#include <volt/plugin/cnat_model.h>
#include <volt/plugin/phase_field_mesh.h>
#include <volt/plugin/lattice_matcher.h>
#include <volt/plugin/structure_contract.h>
#include <volt/analysis/nearest_neighbor_finder.h>
#include <volt/cli/common.h>
#include <volt/core/analysis_result.h>
#include <volt/core/frame_adapter.h>
#include <volt/core/progress.h>
#include <volt/plugin/output_files.h>
#include <volt/plugin/output_serializer.h>
#include <volt/utilities/json_utils.h>
#include <volt/utilities/parquet_atom_writer.h>
#include <volt/utilities/parquet_mesh_writer.h>
#include <volt/structures/crystal_topology_registry.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <utility>

namespace Volt {

using LatticeMatcherDetail::lowercase;

std::vector<std::string> CnatService::classNames(const CnatModel& model) {
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

std::string CnatService::modelName(const CnatModel& model) {
    const json meta = json::parse(model.metadata(), nullptr, false);
    if (meta.is_object() && meta.contains("name") && meta["name"].is_string())
        return meta["name"].get<std::string>();
    return "unnamed";
}

std::optional<std::vector<int>> CnatService::parseTypes(const std::string& text) {
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

std::vector<int> CnatService::structureIds(const std::vector<std::string>& names) {
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

std::vector<CnatService::ClassCrystal> CnatService::modelCrystals(const CnatModel& model, const std::vector<std::string>& names) {
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

int CnatService::contractSpecies(const CellAtoms& cell) {
    std::map<int, int> counts;
    for (const int type : cell.types) counts[type]++;
    int species = -1;
    for (const auto& [type, count] : counts)
        if (species < 0 || count < counts[species]) species = type;
    return species;
}

std::vector<std::string> CnatService::modelTopologies(const std::vector<std::string>& names,
                                                     const std::vector<ClassCrystal>& crystals) {
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

void CnatService::writeProbabilities(ColumnarRowWriter& w, const float* p, const std::vector<std::string>& names) {
    w.field("probability_crystalline", 1.0 - static_cast<double>(p[0]));
    for (std::size_t c = 0; c < names.size(); ++c)
        w.field("probability_" + lowercase(names[c]), static_cast<double>(p[c]));
}

template<typename PointAt>
std::vector<float> CnatService::classifyPoints(const CnatModel& model, const NearestNeighborFinder& finder,
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

std::vector<int> CnatService::assignLabels(const std::vector<float>& probabilities, int classes, double threshold) {
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

std::array<int, 3> CnatService::phaseFieldDims(const SimulationCell& cell, double spacing) {
    std::array<int, 3> dims{};
    for (int d = 0; d < 3; ++d)
        dims[d] = std::max(2, static_cast<int>(std::lround(cell.matrix().column(d).length() / spacing)));
    return dims;
}

json CnatService::countRows(const std::vector<std::int64_t>& counts, const std::vector<std::string>& names,
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
    const std::string name = builtIn ? modelName(model) : _modelPath;
    CnatLog log;
    log.dump(LoadedModel{model, name, builtIn, names});

    // The neighbour types default to those the model was trained with (export_model.py --neighbor_types).
    std::string typeList = _types;
    if (typeList.empty()) {
        const json meta = json::parse(model.metadata(), nullptr, false);
        if (meta.is_object() && meta.contains("neighbor_types") && meta["neighbor_types"].is_array())
            for (const auto& t : meta["neighbor_types"]) typeList += (typeList.empty() ? "" : ",") + std::to_string(t.get<int>());
        if (!typeList.empty()) log.dump(TrainedNeighbourTypes{typeList});
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
        log.dump(NeighbourSublattice{sublattice, frame, typeList});
        cloud = &sublattice;
    }
    if (cloud->natoms < cfg.neighbors)
        return AnalysisResult::failure("The neighbour atoms are fewer than the model's " + std::to_string(cfg.neighbors));

    auto positions = FrameAdapter::createPositionPropertyShared(*cloud);
    if (!positions)
        return AnalysisResult::failure("Failed to create position property");

    // Weighed by this run's atoms and grid points: on the sample runs the model costs about the
    // same per atom and per grid point, the contract a twentieth of that and the index and outputs
    // a hundredth.
    const double atomWeight = static_cast<double>(frame.natoms);
    double gridWeight = 0.0;
    if (_gridSpacing > 0.0) {
        const std::array<int, 3> dims = phaseFieldDims(frame.simulationCell, _gridSpacing);
        gridWeight = static_cast<double>(dims[0]) * dims[1] * dims[2];
    }
    const bool contractRequested = !outputBase.empty() && lowercase(_topologies) != "none";
    ProgressSteps steps({
        {"Building the nearest-neighbour index", 0.01 * atomWeight},
        {"Classifying atoms", atomWeight},
        {"Writing the structure contract", 0.05 * atomWeight, contractRequested},
        {"Evaluating the phase field", gridWeight},
        {"Writing outputs", 0.01 * atomWeight},
    });
    NearestNeighborFinder finder(cfg.neighbors);
    if (!finder.prepare(positions.get(), frame.simulationCell))
        return AnalysisResult::failure("NearestNeighborFinder::prepare failed");
    log.dump(finder);

    steps.next();
    const std::size_t natoms = static_cast<std::size_t>(frame.natoms);
    const std::vector<float> atomProbabilities = classifyPoints(model, finder, natoms,
        [&](std::size_t i) { return frame.positions[i]; }, "Classifying atoms");
    const std::vector<int> atomLabels = assignLabels(atomProbabilities, cfg.classes, _threshold);
    std::vector<std::int64_t> atomCounts(cfg.classes, 0);
    for (int label : atomLabels) atomCounts[label]++;
    log.dump(AtomClasses{atomCounts, names, natoms});

    steps.next();

    json result = AnalysisResult::success();
    result["main_listing"] = {
        {"total_atoms", frame.natoms},
        {"model", name},
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
            requested = automatic ? modelTopologies(names, crystals) : CLI::splitList(_topologies, ',');
            topologies = loadLatticeTopologies(requested);
            log.dump(TopologyRequest{requested, !_cell.empty() || std::ranges::any_of(crystals, [](const ClassCrystal& c) { return c.cell.has_value(); })});
        } catch (const std::exception& e) {
            if (!automatic)
                return AnalysisResult::failure(std::string("Cannot load the lattice topologies: ") + e.what());
            log.dump(UnloadableTopologies{e.what()});
        }
        for (std::size_t c = 1; c < names.size() && automatic; ++c) {
            if (!crystals[c].cell) continue;
            const int species = contractSpecies(*crystals[c].cell);
            try {
                CellTopologies cell = cellTopologies(*crystals[c].cell, lowercase(names[c]), {species}, 0.0,
                    kCustomTopologyTypeBase + static_cast<int>(topologies.size()));
                log.dump(ModelCrystal{names[c], cell, species});
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
                log.dump(ReferenceCell{cell, _cellTypes});
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
            if (auto error = writeStructureContract(frame, outputBase, sites, preferred, topologies, _rmsd, result, contractAtoms, log))
                return AnalysisResult::failure("Cannot write the structure contract: " + *error);
        }
    }
    result["main_listing"]["structure_contract"] = !topologies.empty();
    steps.next();

    if (_gridSpacing > 0.0) {
        const SimulationCell& cell = frame.simulationCell;
        const std::array<int, 3> dims = phaseFieldDims(cell, _gridSpacing);
        const std::size_t gridCount = static_cast<std::size_t>(dims[0]) * dims[1] * dims[2];
        auto gridPoint = [&](std::size_t g) {
            const std::size_t i = g % dims[0];
            const std::size_t j = (g / dims[0]) % dims[1];
            const std::size_t k = g / (static_cast<std::size_t>(dims[0]) * dims[1]);
            return cell.reducedToAbsolute(gridPointReduced(dims, static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)));
        };

        const std::vector<float> gridProbabilities = classifyPoints(model, finder, gridCount, gridPoint, "Evaluating the phase field");
        const std::vector<int> gridLabels = assignLabels(gridProbabilities, cfg.classes, _threshold);
        std::vector<std::int64_t> gridCounts(cfg.classes, 0);
        for (int label : gridLabels) gridCounts[label]++;
        log.dump(PhaseFieldClasses{gridCounts, names, gridCount});

        result["main_listing"]["grid"] = std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + "x" + std::to_string(dims[2]);
        result["main_listing"]["grid_spacing"] = _gridSpacing;
        result["sub_listings"]["volume_fractions"] = countRows(gridCounts, names, ids, gridCount, "grid_points");

        if (_exportInterfaceMesh) {
            std::vector<float> field(gridCount);
            for (std::size_t g = 0; g < gridCount; ++g) field[g] = 1.0f - gridProbabilities[g * cfg.classes];
            if (_interfaceSmoothing > 0.0)
                field = smoothField(field, dims, cell, _interfaceSmoothing);
            const PhaseFieldMesh mesh = extractIsosurface(field, dims, cell, static_cast<float>(_threshold));
            log.dump(mesh);
            result["main_listing"]["interface_area"] = mesh.area;
            if (!outputBase.empty()) {
                const std::string meshPath = outputPath(outputBase, "interface_mesh");
                json meshDoc = writeMeshTables(meshPath, mesh.vertices, mesh.facets, cell);
                meshDoc["main_listing"]["area"] = mesh.area;
                JsonUtils::writeJsonToParquet(meshDoc, meshPath);
            }
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
            streamAtomsToParquet(outputPath(outputBase, "phase_field"), grid,
                [&](std::size_t g) { return names[gridLabels[g]]; },
                [&](ColumnarRowWriter& w, std::size_t g) {
                    writeProbabilities(w, gridProbabilities.data() + g * cfg.classes, names);
                },
                [&](std::size_t g) { return ids[gridLabels[g]]; },
                true);
        }
    }
    steps.next();

    if (!outputBase.empty()) {
        Plugin::serializePluginOutput(outputBase, frame, result, {
            .summaryFileSuffix = "_cnat",
            .bucketResolver = [&](std::size_t i) { return names[atomLabels[i]]; },
            .perAtomColumnWriter = [&](ColumnarRowWriter& w, std::size_t i) {
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
