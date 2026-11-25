#pragma once

#include <volt/analysis/structure_identification_log.h>
#include <volt/plugin/cnat_service.h>
#include <volt/plugin/structure_contract.h>

namespace Volt {

struct PhaseFieldMesh;

// The execution log of one CNAT run, on top of the lines every structure identification prints. Each phase hands it
// what it built, and the log writes what that holds and how long the phase took. Scripts parse these lines, so their
// wording stays as it is.
class CnatLog : public StructureIdentificationLog {
public:
    using StructureIdentificationLog::dump;

    void dump(const CnatService::LoadedModel& model);
    void dump(const CnatService::TrainedNeighbourTypes& types);
    void dump(const CnatService::NeighbourSublattice& sublattice);
    // The atom classes, with how long the model took per atom.
    void dump(const CnatService::AtomClasses& classes);
    void dump(const CnatService::TopologyRequest& request);
    void dump(const CnatService::UnloadableTopologies& failure);
    void dump(const CnatService::ModelCrystal& crystal);
    void dump(const CnatService::ReferenceCell& cell);

    // The alignment's line waits for the grains grown from it, which move its sites; until then the log keeps how
    // long it took.
    void dump(const ContractSiteAlignment& alignment);
    void dump(const ContractGrainGrowth& growth);
    void dump(const ContractClusterRules& rules);

    void dump(const CnatService::PhaseFieldClasses& classes);
    // The interface mesh extracted from the phase field.
    void dump(const PhaseFieldMesh& mesh);

private:
    long long _alignmentMilliseconds = 0;
};

}
