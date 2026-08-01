cmake_minimum_required(VERSION 3.20)
project(openntc LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_FLAGS_RELWITHDEBINFO MATCHES "-lineinfo")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -lineinfo")
endif()

add_executable(openntc openntc/openntc.cpp openntc/ntc_kernel.cu openntc/dataloader.cpp)
set_target_properties(openntc PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libcmt.lib")
set_target_properties(openntc PROPERTIES CUDA_ARCHITECTURES native)
target_include_directories(openntc PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(openntc PRIVATE CUDA::cublas)

add_executable(test_backward openntc/test_backward.cpp openntc/ntc_kernel.cu)
set_target_properties(test_backward PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libcmt.lib")
set_target_properties(test_backward PROPERTIES CUDA_ARCHITECTURES native)
target_include_directories(test_backward PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_backward PRIVATE CUDA::cublas)

add_executable(test_forward openntc/test_forward.cpp openntc/ntc_kernel.cu)
set_target_properties(test_forward PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libcmt.lib")
set_target_properties(test_forward PROPERTIES CUDA_ARCHITECTURES native)
target_include_directories(test_forward PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_forward PRIVATE CUDA::cublas)

add_executable(test_dataloader openntc/test_dataloader.cpp openntc/dataloader.cpp)
set_target_properties(test_dataloader PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libcmt.lib")
target_include_directories(test_dataloader PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_dataloader PRIVATE CUDA::cudart)