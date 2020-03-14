cmake_minimum_required(VERSION 3.8.2)
project(Sample_0)
set(PROJ_NAME sample_0)
add_executable(${PROJ_NAME}
    ./sample_0.cu
)
enable_language(CUDA)
set_property(TARGET ${PROJ_NAME} PROPERTY CUDA_STANDARD 11)
