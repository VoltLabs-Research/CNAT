#include <volt/plugin/lattice_matcher.h>
#include <volt/analysis/crystal_symmetry_utils.h>
#include <volt/structures/crystal_topology_registry.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace Volt {

namespace {

// A neighbour pair is tried as the image of the anchor pair when its lengths and angle cosine are this
// close to the anchors'; the bounds only prune, the assignment residual decides.
constexpr double kAnchorLengthTolerance = 0.3;
constexpr double kAnchorCosineTolerance = 0.35;
constexpr int kRefinementPasses = 3;
constexpr int kPolarIterations = 30;

std::string lowercase(std::string s) {
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// Orthonormal frame (columns) with its first axis along a and its second in the a-b plane.
bool anchorFrame(const Vector3& a, const Vector3& b, Matrix3& frame) {
    const Vector3 normal = a.cross(b);
    const double normalLength = normal.length();
    if (!(normalLength > 1e-6 * a.length() * b.length())) return false;
    const Vector3 e1 = a / a.length();
    const Vector3 e3 = normal / normalLength;
    frame = Matrix3(e1, e3.cross(e1), e3);
    return true;
}

// Each scaled, rotated ideal vector takes its nearest neighbour of the candidates (of its type, with
// types); false when a neighbour is taken twice or a vector finds none.
bool assign(const LatticeTopology& topology, const Matrix3& rotation, double scale, const Vector3* v, int candidates,
            const int* types, std::array<int, MAX_NEIGHBORS>& slots, double& residual) {
    const int cn = topology.coordinationNumber;
    std::uint32_t taken = 0;
    residual = 0.0;
    for (int k = 0; k < cn; ++k) {
        const Vector3 ideal = scale * (rotation * topology.unitVectors[k]);
        int nearest = -1;
        double nearestDistance = std::numeric_limits<double>::max();
        for (int j = 0; j < candidates; ++j) {
            if (types && types[j] != topology.neighborTypes[k]) continue;
            const double distance = (v[j] - ideal).squaredLength();
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearest = j;
            }
        }
        if (nearest < 0 || (taken & (1u << nearest))) return false;
        taken |= 1u << nearest;
        slots[k] = nearest;
        residual += nearestDistance;
    }
    return true;
}

// Rotation factor of the polar decomposition by Newton's iteration X <- (X + X^-T) / 2, where the
// columns of X^-T are the cross products of X's columns over det X. False unless det m > 0, i.e. unless
// the best orthogonal fit is a proper rotation.
bool rotationFactor(Matrix3 m, Matrix3& rotation) {
    if (!(m.determinant() > 1e-12)) return false;
    for (int i = 0; i < kPolarIterations; ++i) {
        const double det = m.determinant();
        const Matrix3 inverseTransposed(m.column(1).cross(m.column(2)) / det,
                                        m.column(2).cross(m.column(0)) / det,
                                        m.column(0).cross(m.column(1)) / det);
        const Matrix3 next = 0.5 * (m + inverseTransposed);
        double change = 0.0;
        for (int c = 0; c < 3; ++c) change += (next.column(c) - m.column(c)).squaredLength();
        m = next;
        if (change < 1e-24) break;
    }
    rotation = m;
    return true;
}

// The deltas in units of their mean length over the shell, and how many of them are candidates: the
// shell for a topology without neighbour types, else every delta given (matchLattice).
bool normalizeDeltas(const LatticeTopology& topology, const Vector3* deltas, const int* deltaTypes, int deltaCount,
                     std::array<Vector3, kMaxMatchCandidates>& v, int& candidates, const int*& types) {
    const int cn = topology.coordinationNumber;
    types = deltaTypes && !topology.neighborTypes.empty() ? deltaTypes : nullptr;
    candidates = types ? std::min(deltaCount, kMaxMatchCandidates) : cn;
    if (candidates < cn) return false;
    double meanLength = 0.0;
    for (int j = 0; j < cn; ++j) meanLength += deltas[j].length();
    meanLength /= cn;
    if (!(meanLength > 0.0)) return false;
    for (int j = 0; j < candidates; ++j) v[j] = deltas[j] / meanLength;
    return true;
}

// Least-squares rotation and scale for the assignment, then the assignment again under them; each
// step can only lower the residual.
void refine(const LatticeTopology& topology, const Vector3* v, int candidates, const int* types,
            std::array<int, MAX_NEIGHBORS>& slots, double& bestResidual, Matrix3& bestRotation) {
    const int cn = topology.coordinationNumber;
    std::array<int, MAX_NEIGHBORS> candidate{};
    for (int pass = 0; pass < kRefinementPasses; ++pass) {
        Matrix3 covariance(Matrix3::Zero{});
        for (int k = 0; k < cn; ++k)
            for (int c = 0; c < 3; ++c) covariance.column(c) += v[slots[k]] * topology.unitVectors[k][c];
        Matrix3 rotation;
        if (!rotationFactor(covariance, rotation)) break;
        double projected = 0.0;
        double idealNorm = 0.0;
        for (int k = 0; k < cn; ++k) {
            const Vector3 ideal = rotation * topology.unitVectors[k];
            projected += v[slots[k]].dot(ideal);
            idealNorm += ideal.squaredLength();
        }
        double residual = 0.0;
        if (!assign(topology, rotation, projected / idealNorm, v, candidates, types, candidate, residual)) break;
        const bool converged = std::equal(slots.begin(), slots.begin() + cn, candidate.begin());
        slots = candidate;
        bestResidual = residual;
        bestRotation = rotation;
        if (converged) break;
    }

}

// Unit shell and anchor pair of a topology whose shell is set.
void prepareMatching(LatticeTopology& topology, const std::vector<Vector3>& shell) {
    const int cn = topology.coordinationNumber;
    double meanLength = 0.0;
    for (int k = 0; k < cn; ++k) meanLength += shell[k].length();
    meanLength /= cn;
    for (int k = 0; k < cn; ++k) topology.unitVectors.push_back(shell[k] / meanLength);

    // The shortest vectors move least under thermal motion; of them, the pair closest to
    // perpendicular gives the best-conditioned frame. Other vectors join only when the shortest
    // are all collinear.
    const auto& u = topology.unitVectors;
    double shortest = std::numeric_limits<double>::max();
    for (const Vector3& vector : u) shortest = std::min(shortest, vector.length());
    auto isShortest = [&](int k) { return u[k].length() <= shortest * (1.0 + 1e-6); };
    topology.anchorA = 0;
    while (!isShortest(topology.anchorA)) ++topology.anchorA;
    const Vector3& a = u[topology.anchorA];
    for (const bool shortestOnly : {true, false}) {
        double bestCosine = 1.0 - 1e-6;
        topology.anchorB = -1;
        for (int k = 0; k < cn; ++k) {
            if (k == topology.anchorA || (shortestOnly && !isShortest(k))) continue;
            const double cosine = std::abs(a.dot(u[k])) / (a.length() * u[k].length());
            if (cosine < bestCosine) {
                bestCosine = cosine;
                topology.anchorB = k;
            }
        }
        if (topology.anchorB >= 0) break;
    }
    Matrix3 frame;
    if (topology.anchorB < 0 || !anchorFrame(a, u[topology.anchorB], frame))
        throw std::runtime_error("lattice topology '" + topology.name + "' has no two non-collinear neighbour vectors");
    topology.anchorFrameTransposed = frame.transposed();
}

}

// LAMMPS data file, orthogonal or triclinic, 'atomic' (id type x y z) or 'charge' (id type q x y z) style.
CellAtoms readLammpsData(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open the reference cell " + path);
    CellAtoms data;
    std::array<double, 3> lo{}, hi{}, tilt{};
    std::size_t natoms = 0;
    std::string line;
    bool charge = false;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        std::vector<std::string> tok;
        for (std::string w; words >> w;) tok.push_back(w);
        if (tok.size() >= 2 && tok[1] == "atoms") natoms = std::stoul(tok[0]);
        for (int d = 0; d < 3; ++d)
            if (tok.size() >= 4 && tok[2] == std::string(1, "xyz"[d]) + "lo") {
                lo[d] = std::stod(tok[0]);
                hi[d] = std::stod(tok[1]);
            }
        if (tok.size() >= 6 && tok[3] == "xy" && tok[4] == "xz" && tok[5] == "yz")
            for (int d = 0; d < 3; ++d) tilt[d] = std::stod(tok[d]);
        if (!tok.empty() && tok[0] == "Atoms") {
            charge = line.find("charge") != std::string::npos;
            break;
        }
    }
    if (natoms == 0) throw std::runtime_error("the reference cell " + path + " has no atoms or no Atoms section");
    data.cell = Matrix3(Vector3(hi[0] - lo[0], 0, 0), Vector3(tilt[0], hi[1] - lo[1], 0), Vector3(tilt[1], tilt[2], hi[2] - lo[2]));
    data.origin = Point3(lo[0], lo[1], lo[2]);
    while (data.positions.size() < natoms && std::getline(in, line)) {
        std::istringstream words(line);
        std::vector<double> v;
        for (double x; words >> x;) v.push_back(x);
        if (v.size() < 5) continue;
        // Without a style comment, 6 or 9 columns (with image flags) mean a charge column.
        const bool hasCharge = charge || v.size() == 6 || v.size() == 9;
        const std::size_t x = hasCharge ? 3 : 2;
        if (v.size() < x + 3) throw std::runtime_error("short atom line in " + path + ": " + line);
        data.types.push_back(static_cast<int>(v[1]));
        data.positions.emplace_back(v[x], v[x + 1], v[x + 2]);
    }
    if (data.positions.size() != natoms) throw std::runtime_error("the reference cell " + path + " lists fewer atoms than it declares");
    return data;
}

int structureTypeOfName(const std::string& name) {
    const std::string key = lowercase(name);
    for (int type = StructureType::SC; type < StructureType::NUM_STRUCTURE_TYPES; ++type)
        if (key == lowercase(structureTypeName(type))) return type;
    return StructureType::OTHER;
}

std::vector<LatticeTopology> loadLatticeTopologies(const std::vector<std::string>& names) {
    std::vector<LatticeTopology> topologies;
    for (const std::string& requested : names) {
        const CrystalTopologyEntry* entry = crystalTopologyByName(requested);
        if (!entry) {
            std::string known;
            for (const auto& e : crystalTopologyRegistry().entries()) known += (known.empty() ? "" : ", ") + e.name;
            throw std::runtime_error("unknown lattice topology '" + requested + "' (known: " + known + ")");
        }
        if (std::ranges::any_of(topologies, [&](const LatticeTopology& t) { return t.name == entry->name; }))
            continue;

        LatticeTopology topology;
        topology.name = entry->name;
        const int cn = entry->coordinationNumber;
        if (cn < 3 || cn > MAX_NEIGHBORS)
            throw std::runtime_error("lattice topology '" + topology.name + "' has " + std::to_string(cn) +
                                     " neighbours; the contract holds 3 to " + std::to_string(MAX_NEIGHBORS));
        topology.coordinationNumber = cn;
        topology.latticeVectors = entry->neighborVectors;
        topology.commonNeighbors = sharedCrystalTopology(entry->name)->commonNeighbors;

        // The rotations carry the shell onto all the lattice vectors, not only onto the shell: in hcp
        // and the diamonds the vectors past the shell belong to the second sublattice, and the rotations
        // between the sublattices (60 degrees about c in hcp) are what join them into one cluster. CNA
        // builds its lattice structures the same way; the registry's own symmetries stop at the shell.
        const std::vector<Vector3> shell(entry->neighborVectors.begin(), entry->neighborVectors.begin() + cn);
        AnalysisSymmetryUtils::generateSymmetryPermutations(shell, cn, topology.latticeVectors, topology.symmetries);
        AnalysisSymmetryUtils::retainProperRotations(topology.symmetries);
        AnalysisSymmetryUtils::calculateSymmetryProducts(topology.symmetries);
        if (topology.symmetries.empty() || topology.symmetries.size() > 63)
            throw std::runtime_error("lattice topology '" + topology.name + "' has " +
                                     std::to_string(topology.symmetries.size()) + " proper rotations; the cluster builder takes 1 to 63");
        const int known = structureTypeOfName(topology.name);
        topology.structureType = known != StructureType::OTHER
            ? known
            : kCustomTopologyTypeBase + static_cast<int>(topologies.size());

        auto isLatticeVector = [&](const Vector3& vector) {
            return std::ranges::any_of(topology.latticeVectors, [&](const Vector3& l) { return (l - vector).isZero(1e-6); });
        };
        std::vector<Vector3> grownBonds;
        for (int k = 0; k < cn; ++k)
            if (std::ranges::all_of(topology.commonNeighbors[k], [&](int c) { return c >= 0 && isLatticeVector(shell[c] - shell[k]); }))
                grownBonds.push_back(shell[k]);
        bool spansSpace = false;
        for (std::size_t i = 0; i < grownBonds.size() && !spansSpace; ++i)
            for (std::size_t j = i + 1; j < grownBonds.size() && !spansSpace; ++j)
                for (std::size_t k = j + 1; k < grownBonds.size() && !spansSpace; ++k)
                    spansSpace = std::abs(Matrix3(grownBonds[i], grownBonds[j], grownBonds[k]).determinant()) > 1e-9;
        topology.orientationClusters = !spansSpace;
        for (std::size_t g = 0; g < topology.symmetries.size(); ++g) {
            const auto& permutation = topology.symmetries[g].permutation;
            if (std::all_of(permutation.begin(), permutation.begin() + cn, [&](int k) { return k >= 0 && k < cn; }))
                topology.shellSymmetries.push_back(static_cast<int>(g));
        }
        prepareMatching(topology, shell);
        topologies.push_back(std::move(topology));
    }
    return topologies;
}

CellTopologies loadCellTopologies(const std::string& path, const std::vector<int>& types, double cutoff,
                                  int firstStructureType) {
    std::string name = std::filesystem::path(path).stem().string();
    for (auto& ch : name) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return cellTopologies(readLammpsData(path), std::move(name), types, cutoff, firstStructureType);
}

CellTopologies cellTopologies(const CellAtoms& data, std::string name, const std::vector<int>& types, double cutoff,
                              int firstStructureType) {
    // A site with fewer neighbours gives circuits too few bonds to close; MAX_NEIGHBORS is the contract's.
    constexpr int kMinimumSiteNeighbors = 12;
    const std::string path = "of " + name;
    std::vector<Vector3> frac;
    std::vector<int> siteTypes;
    const Matrix3 inverse = data.cell.inverse();
    for (std::size_t i = 0; i < data.positions.size(); ++i)
        if (types.empty() || std::ranges::find(types, data.types[i]) != types.end()) {
            frac.push_back(inverse * (data.positions[i] - data.origin));
            siteTypes.push_back(data.types[i]);
        }
    if (frac.empty()) throw std::runtime_error("the reference cell " + path + " has no atoms of the requested types");

    // Neighbours of each site out to a radius that holds more than MAX_NEIGHBORS of them.
    const double volume = std::abs(data.cell.determinant());
    std::array<int, 3> images{};
    std::vector<std::vector<Vector3>> neighbors(frac.size());
    for (double radius = 6.0;; radius *= 1.5) {
        for (int d = 0; d < 3; ++d) {
            const double width = volume / data.cell.column((d + 1) % 3).cross(data.cell.column((d + 2) % 3)).length();
            images[d] = static_cast<int>(std::ceil(radius / width)) + 1;
        }
        bool enough = true;
        for (std::size_t s = 0; s < frac.size(); ++s) {
            neighbors[s].clear();
            for (std::size_t t = 0; t < frac.size(); ++t)
                for (int i = -images[0]; i <= images[0]; ++i)
                    for (int j = -images[1]; j <= images[1]; ++j)
                        for (int k = -images[2]; k <= images[2]; ++k) {
                            const Vector3 delta = data.cell * (frac[t] - frac[s] + Vector3(i, j, k));
                            const double length = delta.length();
                            if (length > 1e-6 && length < radius) neighbors[s].push_back(delta);
                        }
            std::ranges::sort(neighbors[s], [](const Vector3& a, const Vector3& b) {
                const double la = a.squaredLength(), lb = b.squaredLength();
                if (std::abs(la - lb) > 1e-9) return la < lb;
                return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
            });
            enough = enough && neighbors[s].size() > static_cast<std::size_t>(MAX_NEIGHBORS);
        }
        if (enough) break;
    }

    CellTopologies cell;
    cell.name = std::move(name);
    auto neighborCount = [&](std::size_t s, double radius) {
        return static_cast<int>(std::ranges::count_if(neighbors[s], [&](const Vector3& v) { return v.length() < radius; }));
    };
    if (cutoff > 0.0) {
        cell.cutoff = cutoff;
    } else {
        // The widest gap in the pooled neighbour distances that leaves every site 12 to MAX_NEIGHBORS neighbours.
        std::vector<double> distances;
        for (const auto& list : neighbors)
            for (std::size_t n = 0; n <= static_cast<std::size_t>(MAX_NEIGHBORS) && n < list.size(); ++n)
                distances.push_back(list[n].length());
        std::ranges::sort(distances);
        double widest = 0.0;
        for (std::size_t i = 0; i + 1 < distances.size(); ++i) {
            const double gap = distances[i + 1] - distances[i];
            if (gap <= widest) continue;
            const double candidate = 0.5 * (distances[i] + distances[i + 1]);
            bool valid = true;
            for (std::size_t s = 0; s < neighbors.size() && valid; ++s) {
                const int cn = neighborCount(s, candidate);
                valid = cn >= kMinimumSiteNeighbors && cn <= MAX_NEIGHBORS;
            }
            if (!valid) continue;
            widest = gap;
            cell.cutoff = candidate;
        }
        if (cell.cutoff <= 0.0)
            throw std::runtime_error("no neighbour cutoff of the reference cell " + path + " gives every site " +
                                     std::to_string(kMinimumSiteNeighbors) + " to " + std::to_string(MAX_NEIGHBORS) +
                                     " neighbours; pass --cell_cutoff");
    }

    for (std::size_t s = 0; s < neighbors.size(); ++s) {
        const int cn = neighborCount(s, cell.cutoff);
        if (cn < 3 || cn > MAX_NEIGHBORS)
            throw std::runtime_error("site " + std::to_string(s) + " of the reference cell has " + std::to_string(cn) +
                                     " neighbours within " + std::to_string(cell.cutoff) + " A; the contract holds 3 to " +
                                     std::to_string(MAX_NEIGHBORS));
        LatticeTopology topology;
        topology.name = cell.name;
        topology.structureType = firstStructureType + static_cast<int>(s);
        topology.coordinationNumber = cn;
        topology.cellSite = static_cast<int>(s);
        topology.siteType = siteTypes[s];
        for (int k = 0; k < cn; ++k) {
            const Vector3 landed = frac[s] + inverse * neighbors[s][k];
            int site = -1;
            for (std::size_t u = 0; u < frac.size() && site < 0; ++u) {
                Vector3 d = landed - frac[u];
                for (int c = 0; c < 3; ++c) d[c] -= std::round(d[c]);
                if ((data.cell * d).length() < 1e-3) site = static_cast<int>(u);
            }
            topology.neighborTypes.push_back(site >= 0 ? siteTypes[site] : -1);
            topology.neighborSites.push_back(site);
        }
        topology.latticeVectors.assign(neighbors[s].begin(), neighbors[s].begin() + cn);
        for (auto& pair : topology.commonNeighbors) pair = {-1, -1};
        SharedCrystalSymmetryPermutation identity;
        for (int k = 0; k < MAX_NEIGHBORS; ++k) identity.permutation[k] = k;
        identity.inverseProduct = {0};
        topology.symmetries = {identity};
        topology.shellSymmetries = {0};
        prepareMatching(topology, topology.latticeVectors);
        cell.sites.push_back(std::move(topology));
    }

    return cell;
}

bool matchLattice(const LatticeTopology& topology, const Vector3* deltas, LatticeMatch& match,
                  const int* deltaTypes, int deltaCount) {
    const int cn = topology.coordinationNumber;
    const int* types = nullptr;
    int candidates = 0;
    std::array<Vector3, kMaxMatchCandidates> v;
    if (!normalizeDeltas(topology, deltas, deltaTypes, deltaCount, v, candidates, types)) return false;

    const Vector3& a = topology.unitVectors[topology.anchorA];
    const Vector3& b = topology.unitVectors[topology.anchorB];
    const double lengthA = a.length();
    const double lengthB = b.length();
    const double cosineAB = a.dot(b) / (lengthA * lengthB);

    double bestResidual = std::numeric_limits<double>::max();
    std::array<int, MAX_NEIGHBORS> slots{};
    std::array<int, MAX_NEIGHBORS> candidate{};
    Matrix3 bestRotation(Matrix3::Identity{});
    Matrix3 frame;
    for (int p = 0; p < candidates; ++p) {
        if (types && types[p] != topology.neighborTypes[topology.anchorA]) continue;
        const double lengthP = v[p].length();
        if (std::abs(lengthP - lengthA) > kAnchorLengthTolerance * lengthA) continue;
        for (int q = 0; q < candidates; ++q) {
            if (q == p || (types && types[q] != topology.neighborTypes[topology.anchorB])) continue;
            const double lengthQ = v[q].length();
            if (std::abs(lengthQ - lengthB) > kAnchorLengthTolerance * lengthB) continue;
            if (std::abs(v[p].dot(v[q]) / (lengthP * lengthQ) - cosineAB) > kAnchorCosineTolerance) continue;
            if (!anchorFrame(v[p], v[q], frame)) continue;
            double residual = 0.0;
            const Matrix3 rotation = frame * topology.anchorFrameTransposed;
            if (assign(topology, rotation, 1.0, v.data(), candidates, types, candidate, residual) && residual < bestResidual) {
                bestResidual = residual;
                slots = candidate;
                bestRotation = rotation;
            }
        }
    }
    if (bestResidual == std::numeric_limits<double>::max()) return false;

    refine(topology, v.data(), candidates, types, slots, bestResidual, bestRotation);

    // The shared orientation rule compares the orientations of two atoms bond by bond, so every atom of
    // a grain needs the same representative of its symmetry class: the one closest to the identity, as
    // PTM's fundamental zone gives it. R S_g carries ideal vector k where R carries vector p_g[k].
    if (topology.orientationClusters) {
        int closest = -1;
        double closestTrace = -std::numeric_limits<double>::max();
        for (const int g : topology.shellSymmetries) {
            const Matrix3 candidate = bestRotation * topology.symmetries[g].transformation;
            const double trace = candidate(0, 0) + candidate(1, 1) + candidate(2, 2);
            if (trace > closestTrace) {
                closestTrace = trace;
                closest = g;
            }
        }
        if (closest >= 0) {
            const auto& permutation = topology.symmetries[closest].permutation;
            std::array<int, MAX_NEIGHBORS> reordered{};
            for (int k = 0; k < cn; ++k) reordered[k] = slots[permutation[k]];
            slots = reordered;
            bestRotation = bestRotation * topology.symmetries[closest].transformation;
        }
    }

    match.rmsd = std::sqrt(bestResidual / cn);
    match.slots = slots;
    match.rotation = bestRotation;
    return true;
}

bool matchLatticeFrom(const LatticeTopology& topology, const Vector3* deltas, const Matrix3& guess, LatticeMatch& match,
                      const int* deltaTypes, int deltaCount) {
    const int cn = topology.coordinationNumber;
    const int* types = nullptr;
    int candidates = 0;
    std::array<Vector3, kMaxMatchCandidates> v;
    if (!normalizeDeltas(topology, deltas, deltaTypes, deltaCount, v, candidates, types)) return false;
    std::array<int, MAX_NEIGHBORS> slots{};
    double residual = 0.0;
    if (!assign(topology, guess, 1.0, v.data(), candidates, types, slots, residual)) return false;
    Matrix3 rotation = guess;
    refine(topology, v.data(), candidates, types, slots, residual, rotation);
    match.rmsd = std::sqrt(residual / cn);
    match.slots = slots;
    match.rotation = rotation;
    return true;
}

LatticeCrystalInfo::LatticeCrystalInfo(std::vector<LatticeTopology> topologies)
    : _topologies(std::move(topologies)) {}

const LatticeTopology* LatticeCrystalInfo::find(int structureType) const {
    for (const LatticeTopology& topology : _topologies)
        if (topology.structureType == structureType) return &topology;
    return nullptr;
}

int LatticeCrystalInfo::findClosestSymmetryPermutation(int structureType, const Matrix3& rotation) const {
    const LatticeTopology* topology = find(structureType);
    return topology ? AnalysisSymmetryUtils::findClosestSymmetryPermutation(topology->symmetries, rotation) : 0;
}

int LatticeCrystalInfo::coordinationNumber(int structureType) const {
    const LatticeTopology* topology = find(structureType);
    return topology ? topology->coordinationNumber : 0;
}

int LatticeCrystalInfo::commonNeighborIndex(int structureType, int neighborIndex, int commonNeighborSlot) const {
    const LatticeTopology* topology = find(structureType);
    if (!topology || neighborIndex < 0 || neighborIndex >= topology->coordinationNumber ||
        commonNeighborSlot < 0 || commonNeighborSlot > 1)
        return -1;
    return topology->commonNeighbors[neighborIndex][commonNeighborSlot];
}

int LatticeCrystalInfo::symmetryPermutationCount(int structureType) const {
    const LatticeTopology* topology = find(structureType);
    return topology ? static_cast<int>(topology->symmetries.size()) : 0;
}

int LatticeCrystalInfo::symmetryPermutationEntry(int structureType, int symmetryIndex, int neighborIndex) const {
    const LatticeTopology* topology = find(structureType);
    if (!topology || symmetryIndex < 0 || symmetryIndex >= static_cast<int>(topology->symmetries.size()) ||
        neighborIndex < 0 || neighborIndex >= topology->coordinationNumber)
        return -1;
    return topology->symmetries[symmetryIndex].permutation[neighborIndex];
}

const Matrix3& LatticeCrystalInfo::symmetryTransformation(int structureType, int symmetryIndex) const {
    static const Matrix3 identity(Matrix3::Identity{});
    const LatticeTopology* topology = find(structureType);
    if (!topology || symmetryIndex < 0 || symmetryIndex >= static_cast<int>(topology->symmetries.size()))
        return identity;
    return topology->symmetries[symmetryIndex].transformation;
}

int LatticeCrystalInfo::symmetryInverseProduct(int structureType, int symmetryIndex, int transformationIndex) const {
    const LatticeTopology* topology = find(structureType);
    if (!topology || symmetryIndex < 0 || symmetryIndex >= static_cast<int>(topology->symmetries.size()))
        return 0;
    const std::vector<int>& inverseProduct = topology->symmetries[symmetryIndex].inverseProduct;
    if (transformationIndex < 0 || transformationIndex >= static_cast<int>(inverseProduct.size())) return 0;
    return inverseProduct[transformationIndex];
}

const Vector3& LatticeCrystalInfo::latticeVector(int structureType, int latticeVectorIndex) const {
    static const Vector3 zero = Vector3::Zero();
    const LatticeTopology* topology = find(structureType);
    if (!topology || latticeVectorIndex < 0 || latticeVectorIndex >= static_cast<int>(topology->latticeVectors.size()))
        return zero;
    return topology->latticeVectors[latticeVectorIndex];
}

std::string_view LatticeCrystalInfo::topologyName(int structureType) const {
    const LatticeTopology* topology = find(structureType);
    return topology ? std::string_view(topology->name) : std::string_view("OTHER");
}

}
