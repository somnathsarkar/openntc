cmake_minimum_required(VERSION 3.20)
project(openntc LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_FLAGS_RELWITHDEBINFO MATCHES "-lineinfo")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -lineinfo")
endif()

add_library(libopenntc STATIC libopenntc/libopenntc.cpp
  libopenntc/ntc_kernel.cu)
set_target_properties(libopenntc PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libcmt.lib")
set_target_properties(libopenntc PROPERTIES CUDA_ARCHITECTURES native)
target_include_directories(libopenntc PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(libopenntc PRIVATE CUDA::cublas CUDA::cudart)

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

add_library(imgui STATIC
  C:/Dev/imgui/imgui.cpp
  C:/Dev/imgui/imgui_draw.cpp
  C:/Dev/imgui/imgui_tables.cpp
  C:/Dev/imgui/imgui_widgets.cpp
  C:/Dev/imgui/imgui_demo.cpp
  C:/Dev/imgui/backends/imgui_impl_win32.cpp
  C:/Dev/imgui/backends/imgui_impl_dx12.cpp)
target_include_directories(imgui PUBLIC C:/Dev/imgui C:/Dev/imgui/backends)

add_executable(openntc-gui WIN32 openntc-gui/main.cpp)
option(OPENNTC_COOP "Coop Vector Support" ON)
if(OPENNTC_COOP)
  set(AGILITY_DIR C:/Dev/microsoft.direct3d.d3d12.1.721.3-preview/build/native)
else()
  set(AGILITY_DIR C:/Dev/DirectXAgilitySDK/build/native)     # retail 1.619
endif()
target_include_directories(openntc-gui BEFORE PRIVATE ${AGILITY_DIR}/include)
target_compile_definitions(openntc-gui PRIVATE OPENNTC_COOP=$<BOOL:${OPENNTC_COOP}>)
add_custom_command(TARGET openntc-gui POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:openntc-gui>/D3D12
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AGILITY_DIR}/bin/x64/D3D12Core.dll        $<TARGET_FILE_DIR:openntc-gui>/D3D12
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AGILITY_DIR}/bin/x64/d3d12SDKLayers.dll  $<TARGET_FILE_DIR:openntc-gui>/D3D12)
target_link_directories(openntc-gui PRIVATE C:/Dev/DirectXTex/DirectXTex/Bin/Desktop_2022/x64/Debug)
target_link_libraries(openntc-gui PRIVATE D3D12.lib DXGI.lib D3DCompiler.lib DirectXTex.lib)
target_include_directories(openntc-gui PRIVATE C:/Dev/DirectXTex/)
target_include_directories(openntc-gui PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(openntc-gui PRIVATE imgui libopenntc)