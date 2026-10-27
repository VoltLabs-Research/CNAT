cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(cnat VERSION 1.0.0 LANGUAGES CXX)

if(NOT TARGET coretoolkit::coretoolkit)
    find_package(coretoolkit REQUIRED)
endif()
if(NOT TARGET structure-identification::structure-identification)
    find_package(structure-identification REQUIRED)
endif()
if(NOT TARGET yaml-cpp::yaml-cpp)
    find_package(yaml-cpp REQUIRED)
endif()  # transitive dep of structure-identification (static .a), must be linked into the exe

# Built-in models, selected by file stem with --model; the first one is the default.
set(CNAT_MODELS
    "${CMAKE_CURRENT_SOURCE_DIR}/models/metals.cnat"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ni-fcc-liquid.cnat"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/forsterite-o.cnat"
    CACHE STRING "CNAT models built into the plugin binary")
set(_embedded_model "${CMAKE_CURRENT_BINARY_DIR}/cnat_embedded_models.cpp")
string(REPLACE ";" "|" _model_list "${CNAT_MODELS}")
add_custom_command(
    OUTPUT ${_embedded_model}
    COMMAND ${CMAKE_COMMAND} -DINPUTS=${_model_list} -DOUTPUT=${_embedded_model}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedModel.cmake
    DEPENDS ${CNAT_MODELS} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedModel.cmake
    COMMENT "Embedding CNAT models"
    VERBATIM
)

volt_add_plugin(cnat
    SOURCES
        src/cnat_model.cpp
        src/phase_field_mesh.cpp
        src/lattice_matcher.cpp
        src/structure_contract.cpp
        src/cnat_service.cpp
        ${_embedded_model}
    MAIN_SOURCE src/plugin.cpp
    DEPENDENCIES
        structure-identification::structure-identification
        yaml-cpp::yaml-cpp
    INSTALL_DATA lattices
)

option(CNAT_BUILD_TESTS "Build the checks under tests/" OFF)
if(CNAT_BUILD_TESTS)
    add_executable(cnat-parity-check tests/parity_check.cpp)
    target_link_libraries(cnat-parity-check PRIVATE cnat_lib)
    add_executable(cnat-mesh-check tests/mesh_check.cpp)
    target_link_libraries(cnat-mesh-check PRIVATE cnat_lib)
    add_executable(cnat-contract-check tests/contract_check.cpp)
    target_link_libraries(cnat-contract-check PRIVATE cnat_lib)
endif()
