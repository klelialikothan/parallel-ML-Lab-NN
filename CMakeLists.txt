cmake_minimum_required(VERSION 3.13)  # CMake version check
project(parallel-ML-Lab-NN)           # Create project "parallel-ML-Lab-NN"
set(CMAKE_CXX_STANDARD 17)            # Enable c++17 standard

# Add appropriate tsp.cpp file of project root directory as source file
#set(SOURCE_FILES ./q1/nn.cpp)
#set(SOURCE_FILES ./q2/nn.cpp)
set(SOURCE_FILES ./q3/nn.cpp)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(parallel-ML-Lab-NN ${SOURCE_FILES})

# Add linker flag for math library
#target_link_libraries(parallel-ML-Lab-NN m)

#Add linker flags for math library and OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(parallel-ML-Lab-NN PUBLIC m PUBLIC OpenMP::OpenMP_CXX)
endif()