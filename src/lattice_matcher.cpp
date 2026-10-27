#include <volt/plugin/lattice_matcher.h>
#include <volt/analysis/crystal_symmetry_utils.h>
#include <volt/structures/crystal_topology_registry.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

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

// Each scaled, rotated ideal vector takes its nearest neighbour; false when a neighbour is taken twice.
bool assign(const LatticeTopology& topology, const Matrix3& rotation, double scale, const Vector3* v,
            std::array<int, MAX_NEIGHBORS>& slots, double& residual) {
    const int cn = topology.coordinationNumber;
    std::uint32_t taken = 0;
    residual = 0.0;
    for (int k = 0; k < cn; ++k) {
        const Vector3 ideal = scale * (rotation * topology.unitVectors[k]);
        int nearest = 0;
        double nearestDistance = std::numeric_limits<double>::max();
        for (int j = 0; j < cn; ++j) {
            const double distance = (v[j] - ideal).squaredLength();
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearest = j;
            }
        }
        if (taken & (1u << nearest)) return false;
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
        topologies.push_back(std::move(topology));
    }
    return topologies;
}

bool matchLattice(const LatticeTopology& topology, const Vector3* deltas, LatticeMatch& match) {
    const int cn = topology.coordinationNumber;
    std::array<Vector3, MAX_NEIGHBORS> v;
    double meanLength = 0.0;
    for (int j = 0; j < cn; ++j) meanLength += deltas[j].length();
    meanLength /= cn;
    if (!(meanLength > 0.0)) return false;
    for (int j = 0; j < cn; ++j) v[j] = deltas[j] / meanLength;

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
    for (int p = 0; p < cn; ++p) {
        const double lengthP = v[p].length();
        if (std::abs(lengthP - lengthA) > kAnchorLengthTolerance * lengthA) continue;
        for (int q = 0; q < cn; ++q) {
            if (q == p) continue;
            const double lengthQ = v[q].length();
            if (std::abs(lengthQ - lengthB) > kAnchorLengthTolerance * lengthB) continue;
            if (std::abs(v[p].dot(v[q]) / (lengthP * lengthQ) - cosineAB) > kAnchorCosineTolerance) continue;
            if (!anchorFrame(v[p], v[q], frame)) continue;
            double residual = 0.0;
            const Matrix3 rotation = frame * topology.anchorFrameTransposed;
            if (assign(topology, rotation, 1.0, v.data(), candidate, residual) && residual < bestResidual) {
                bestResidual = residual;
                slots = candidate;
                bestRotation = rotation;
            }
        }
    }
    if (bestResidual == std::numeric_limits<double>::max()) return false;

    // Least-squares rotation and scale for the assignment, then the assignment again under them; each
    // step can only lower the residual.
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
        if (!assign(topology, rotation, projected / idealNorm, v.data(), candidate, residual)) break;
        const bool converged = std::equal(slots.begin(), slots.begin() + cn, candidate.begin());
        slots = candidate;
        bestResidual = residual;
        bestRotation = rotation;
        if (converged) break;
    }

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
