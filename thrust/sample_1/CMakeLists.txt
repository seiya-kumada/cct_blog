cmake_minimum_required(VERSION 3.8.2)
project(Sample_1)
set(PROJ_NAME sample_1)
add_executable(${PROJ_NAME}
    ./sample_1.cu
)
enable_language(CUDA)
find_package(OpenCV REQUIRED)

set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
set(LIBRARIES ${OpenCV_LIBRARIES})

include_directories(${INCLUDE_DIRS})
target_link_libraries(${PROJ_NAME} ${LIBRARIES})

set_property(TARGET ${PROJ_NAME} PROPERTY CUDA_STANDARD 11)

message(STATUS ${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_LIBRARIES})
message(STATUS ${Boost_LIBRARY_DIRS})
