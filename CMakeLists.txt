cmake_minimum_required(VERSION 3.16)

# ── Version from git tag ─────────────────────────────────────────────────────
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --match "v*" --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAG OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(GIT_TAG)
        string(REGEX REPLACE "^v" "" NV_VERSION "${GIT_TAG}")
        string(REGEX REPLACE "[^0-9.].*" "" NV_VERSION "${NV_VERSION}")
    endif()
endif()
if(NOT NV_VERSION)
    set(NV_VERSION "0.0.0")
endif()

project(neroued_vectorizer VERSION ${NV_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

# ── Options ──────────────────────────────────────────────────────────────────
option(NV_BUILD_EVAL "Build the vectorize evaluation library" ON)
option(NV_BUILD_APPS "Build CLI applications" ON)
option(NV_BUILD_TESTS "Build unit tests" ON)
option(NV_BUILD_PYTHON "Build Python bindings (requires pybind11)" OFF)

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs)

include(FetchContent)

# spdlog
if(NOT TARGET spdlog::spdlog_header_only AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog/CMakeLists.txt")
    add_subdirectory(3rdparty/spdlog EXCLUDE_FROM_ALL)
elseif(NOT TARGET spdlog::spdlog_header_only)
    find_package(spdlog QUIET)
    if(NOT spdlog_FOUND)
        set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
        set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.14.1
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(spdlog)
    endif()
endif()

# Clipper2
if(NOT TARGET Clipper2 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/clipper2/CPP/CMakeLists.txt")
    set(CLIPPER2_UTILS OFF CACHE BOOL "" FORCE)
    set(CLIPPER2_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CLIPPER2_TESTS OFF CACHE BOOL "" FORCE)
    set(CLIPPER2_USINGZ "OFF" CACHE STRING "" FORCE)
    add_subdirectory(3rdparty/clipper2/CPP EXCLUDE_FROM_ALL)
    if(TARGET Clipper2)
        target_compile_options(Clipper2 PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/w>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
    endif()
elseif(NOT TARGET Clipper2)
    find_package(Clipper2 QUIET)
    if(NOT Clipper2_FOUND)
        set(CLIPPER2_UTILS OFF CACHE BOOL "" FORCE)
        set(CLIPPER2_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(CLIPPER2_TESTS OFF CACHE BOOL "" FORCE)
        set(CLIPPER2_USINGZ "OFF" CACHE STRING "" FORCE)
        FetchContent_Declare(clipper2
            GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
            GIT_TAG Clipper2_2.0.1
            GIT_SHALLOW TRUE
            SOURCE_SUBDIR CPP)
        FetchContent_MakeAvailable(clipper2)
        if(TARGET Clipper2)
            target_compile_options(Clipper2 PRIVATE
                $<$<CXX_COMPILER_ID:MSVC>:/w>
                $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
        endif()
    endif()
endif()

# Potrace — pass -DPOTRACE_ROOT=<prefix> on Windows; Linux/macOS use system paths
find_library(POTRACE_LIBRARY potrace
    HINTS ${POTRACE_ROOT}/lib ${POTRACE_ROOT})
find_path(POTRACE_INCLUDE_DIR potracelib.h
    HINTS ${POTRACE_ROOT}/include ${POTRACE_ROOT})
if(NOT POTRACE_LIBRARY OR NOT POTRACE_INCLUDE_DIR)
    message(FATAL_ERROR
        "Potrace is required but was not found.\n"
        "  Linux/Debian: sudo apt install libpotrace-dev\n"
        "  macOS:        brew install potrace\n"
        "  Windows:      set POTRACE_ROOT to installation prefix")
endif()

# ── nanosvg (vendored single-header SVG parser) ──────────────────────────────
add_library(nanosvg STATIC 3rdparty/nanosvg/nanosvg_impl.cpp)
target_include_directories(nanosvg SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
target_compile_options(nanosvg PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/w>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)

# ── Main library ─────────────────────────────────────────────────────────────
add_library(neroued_vectorizer STATIC
    src/vectorizer.cpp
    src/logging.cpp
    src/pipeline.cpp
    src/preprocess/preprocess.cpp
    src/segment/color_segment.cpp
    src/segment/slic.cpp
    src/segment/morphology.cpp
    src/boundary/boundary_graph.cpp
    src/boundary/subpixel_refine.cpp
    src/boundary/aa_detector.cpp
    src/contour/assembly.cpp
    src/contour/thin_line.cpp
    src/curve/bezier.cpp
    src/curve/fitting.cpp
    src/trace/potrace.cpp
    src/trace/coverage.cpp
    src/trace/topology.cpp
    src/output/svg_writer.cpp
    src/output/shape_merge.cpp
    src/detail/icc_utils.cpp)

add_library(neroued::vectorizer ALIAS neroued_vectorizer)

target_include_directories(neroued_vectorizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${POTRACE_INCLUDE_DIR})

target_link_libraries(neroued_vectorizer
    PUBLIC
        opencv_core opencv_imgproc opencv_imgcodecs
        spdlog::spdlog_header_only
    PRIVATE
        Clipper2
        ${POTRACE_LIBRARY})

target_compile_definitions(neroued_vectorizer PRIVATE NV_HAS_POTRACE=1)

# Optional: lcms2 + libjpeg for ICC color management
find_library(LCMS2_LIBRARY lcms2)
find_path(LCMS2_INCLUDE_DIR lcms2.h)
if(LCMS2_LIBRARY AND LCMS2_INCLUDE_DIR)
    find_package(JPEG QUIET)
    if(JPEG_FOUND)
        target_include_directories(neroued_vectorizer PRIVATE ${LCMS2_INCLUDE_DIR})
        target_link_libraries(neroued_vectorizer PRIVATE ${LCMS2_LIBRARY} JPEG::JPEG)
        target_compile_definitions(neroued_vectorizer PRIVATE NV_HAS_LCMS2=1)
        message(STATUS "Found lcms2: ${LCMS2_LIBRARY}")
    endif()
endif()

# Optional: OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neroued_vectorizer PUBLIC OpenMP::OpenMP_CXX)
endif()

# Compiler flags
target_compile_options(neroued_vectorizer PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:RelWithDebInfo>>:-O2 -g>)

# ── Eval library ─────────────────────────────────────────────────────────────
if(NV_BUILD_EVAL)
    add_subdirectory(eval)
endif()

# ── CLI apps ─────────────────────────────────────────────────────────────────
if(NV_BUILD_APPS)
    add_subdirectory(apps)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
if(NV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Python bindings ───────────────────────────────────────────────────────────
if(NV_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core python/bindings.cpp)
    target_link_libraries(_core PRIVATE neroued_vectorizer)
    target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    install(TARGETS _core DESTINATION neroued_vectorizer)
endif()

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS neroued_vectorizer
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/neroued
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")
