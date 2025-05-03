# Library definition
add_library(METAL_CPP ${CMAKE_CURRENT_SOURCE_DIR}/definition.cpp)

# Metal cpp headers
target_include_directories(METAL_CPP PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp-extension"
)
target_compile_features(METAL_CPP PRIVATE cxx_std_17)

message(STATUS "Metal Backend")

set(EXTRA_LIBS 
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    "-framework Foundation"
    "-framework QuartzCore")

target_link_libraries(METAL_CPP ${EXTRA_LIBS})