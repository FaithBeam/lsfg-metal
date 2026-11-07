cmake_minimum_required(VERSION 3.20)

# Universal by default: Wine/CrossOver/D3DMetal run as x86_64 under Rosetta 2,
# native Metal apps (and MoltenVK in arm64 Wine builds) run as arm64.
if(NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Target architectures")
endif()
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")
endif()

project(lsfg-metal LANGUAGES C CXX OBJC OBJCXX)

option(LSFGMETAL_BUILD_CLI     "Build the command line tool"    ON)
option(LSFGMETAL_BUILD_TESTS   "Build the test app and tests"   ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_OBJCXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- SPIRV-Cross (SPIR-V -> MSL), namespaced to avoid clashing with MoltenVK's copy in-process
set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SHARED OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_STATIC ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_FORCE_PIC ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_NAMESPACE_OVERRIDE "lsfg_spirv_cross" CACHE STRING "" FORCE)
add_subdirectory(thirdparty/SPIRV-Cross EXCLUDE_FROM_ALL)

set(LSFGMETAL_OBJC_FLAGS -fobjc-arc)

# --- common (configuration, detection, logging)
add_library(lsfg-metal-common STATIC
    src/common/src/config.cpp
    src/common/src/detection.cpp
    src/common/src/log.cpp)
target_include_directories(lsfg-metal-common PUBLIC src/common/include)
target_include_directories(lsfg-metal-common SYSTEM PRIVATE src/common/thirdparty)

# --- backend (Metal port of lsfg-vk-backend)
add_library(lsfg-metal-backend STATIC
    src/backend/src/backend.mm
    src/backend/src/dll_reader.cpp
    src/backend/src/shader.mm
    src/backend/src/shader_registry.mm
    src/backend/src/shaderchains.mm)
target_include_directories(lsfg-metal-backend PUBLIC src/backend/include PRIVATE src/backend/src)
target_compile_options(lsfg-metal-backend PRIVATE ${LSFGMETAL_OBJC_FLAGS})
target_link_libraries(lsfg-metal-backend
    PUBLIC lsfg-metal-common "-framework Metal" "-framework Foundation"
    PRIVATE spirv-cross-msl spirv-cross-glsl spirv-cross-core)

# --- presenter (frame copy / present helpers shared by the hook and the CLI)
add_library(lsfg-metal-presenter STATIC
    src/layer/presenter.mm)
target_include_directories(lsfg-metal-presenter PUBLIC src/layer)
target_compile_options(lsfg-metal-presenter PRIVATE ${LSFGMETAL_OBJC_FLAGS})
target_link_libraries(lsfg-metal-presenter PUBLIC lsfg-metal-backend "-framework QuartzCore" "-framework CoreGraphics")

# --- the injectable hook library
add_library(lsfg-metal SHARED
    src/layer/hooks.mm
    src/layer/capture.mm)
target_compile_options(lsfg-metal PRIVATE ${LSFGMETAL_OBJC_FLAGS})
target_link_libraries(lsfg-metal PRIVATE lsfg-metal-presenter
    "-framework Metal" "-framework QuartzCore" "-framework Foundation"
    "-framework CoreGraphics" "-framework ImageIO")
set_target_properties(lsfg-metal PROPERTIES
    OUTPUT_NAME "lsfg-metal"
    INSTALL_NAME_DIR "@rpath")

add_custom_command(TARGET lsfg-metal POST_BUILD
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_initializers.sh" "$<TARGET_FILE:lsfg-metal>"
    COMMENT "Checking that liblsfg-metal has no static initializers")

install(TARGETS lsfg-metal LIBRARY DESTINATION lib)
install(PROGRAMS scripts/lsfg-metal-run scripts/lsfg-metal-crossover scripts/lsfg-metal-sikarugir DESTINATION bin)

if(LSFGMETAL_BUILD_CLI)
    add_executable(lsfg-metal-cli src/cli/main.mm)
    target_compile_options(lsfg-metal-cli PRIVATE ${LSFGMETAL_OBJC_FLAGS})
    target_link_libraries(lsfg-metal-cli PRIVATE lsfg-metal-presenter
        "-framework CoreGraphics" "-framework ImageIO")
    target_include_directories(lsfg-metal-cli PRIVATE src/backend/src)
    install(TARGETS lsfg-metal-cli RUNTIME DESTINATION bin)
endif()

if(LSFGMETAL_BUILD_TESTS)
    add_executable(lsfg-metal-testapp tests/testapp/main.mm)
    target_compile_options(lsfg-metal-testapp PRIVATE ${LSFGMETAL_OBJC_FLAGS})
    target_link_libraries(lsfg-metal-testapp PRIVATE
        "-framework Cocoa" "-framework Metal" "-framework QuartzCore")
endif()
