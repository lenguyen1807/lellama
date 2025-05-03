set(LIB_NAME legrad)
set(SHARED_LIB_NAME ${LIB_NAME}_shared)

# --- Source and Include File Gathering ---
file(GLOB_RECURSE LEGRAD_SRC_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp"
)

file(GLOB_RECURSE LEGRAD_INCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/macros/*.h"
)

set(INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/"
)

# --- Options ---
# Debug level
option(LEGRAD_DEBUG_1 "legrad: Set debug at level 1 (Info)" OFF)
option(LEGRAD_DEBUG_2 "legrad: Set debug at level 2 (Debug)" OFF)
option(LEGRAD_DEBUG_3 "legrad: Set debug at level 3 (Trace)" OFF)
option(LEGRAD_WARNINGS "legrad: Enable Warnings" ON) # Enable compiler warnings by default.

if (LEGRAD_DEBUG_1)
    message(STATUS "Set Debug to level 1 (Info)")
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_DEBUG_1")
endif()

if (LEGRAD_DEBUG_2)
    message(STATUS "Set Debug to level 2 (Debug)")
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_DEBUG_2")
endif()

if (LEGRAD_DEBUG_3)
    message(STATUS "Set Debug to level 3 (Trace)")
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_DEBUG_3")
endif()

# Build options for tests, examples, and benchmarks.
option(LEGRAD_BUILD_TESTS "legrad: Build tests" OFF)

# --- External Library Handling ---
# Add OpenCV
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()
message(STATUS "Use OpenCV")
set(LEGRAD_EXTRA_LIBS ${LEGRAD_EXTRA_LIBS} ${OpenCV_LIBS})

# Add fmt
include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281
)
FetchContent_MakeAvailable(fmt)
message(STATUS "Use fmt")
set(LEGRAD_EXTRA_LIBS ${LEGRAD_EXTRA_LIBS} fmt::fmt)

# Use threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Use Boost
find_package(Boost CONFIG COMPONENTS system log log_setup REQUIRED)

# Add Metal and Accelerate
if (APPLE)
    # Copy shader to binary directory
    file(GLOB SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/backend/kernels/*")
    foreach(SHADER_FILE ${SHADER_FILES})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
        configure_file(${SHADER_FILE} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SHADER_NAME}" COPYONLY)
    endforeach()

    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/backend/metal")
    set(LEGRAD_EXTRA_LIBS ${LEGRAD_EXTRA_LIBS} METAL_CPP)

    set(LEGRAD_SRC_FILES ${LEGRAD_SRC_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal_mgr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal_allocator.cpp
    )

    set(LEGRAD_INCLUDE_FILES ${LEGRAD_INCLUDE_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal_mgr.h
        ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal_allocator.h
    )
endif()

# Add Sanitizer
add_compile_options(-fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined)
link_libraries(-fsanitize=address -fsanitize=undefined)

# Optimization for Apple M
add_compile_options(-mcpu=native)

# Compiler Warnings
set(cxx_flags # Common and useful compiler warning flags.
    -Wall # Enable all common warnings.
    -Wextra # Enable extra warnings.
    -Wpedantic # Enable pedantic warnings (stricter standard conformance).
    -Wcast-qual # Warn about casting away qualifiers (const, volatile).
    -Wno-unused-function # Suppress warnings about unused functions (can be noisy in header-only libraries or during development).
) 
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>")

# Add Profiller
add_compile_options(-pg)

# Link Time Optimization (LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if (result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO is not supported: ${output}")
endif()

# --- Library Creation (Shared) ---
message(STATUS "Build shared library")
add_library(${SHARED_LIB_NAME} SHARED ${LEGRAD_SRC_FILES})
target_include_directories(${SHARED_LIB_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
    $<INSTALL_INTERFACE:${INCLUDE_DIR}>
    ${LEGRAD_INCLUDE_FILES}
)
target_link_libraries(${SHARED_LIB_NAME} PUBLIC 
                    Boost::system
                    Threads::Threads
                    ${LEGRAD_EXTRA_LIBS})
target_compile_definitions(${SHARED_LIB_NAME} PUBLIC KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/backend/kernels/" ${LEGRAD_COMPILE_DEFINITIONS})
target_compile_features(${SHARED_LIB_NAME} PRIVATE cxx_std_17)
