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

add_library(imgui STATIC
  C:/Dev/imgui/imgui.cpp
  C:/Dev/imgui/imgui_draw.cpp
  C:/Dev/imgui/imgui_tables.cpp
  C:/Dev/imgui/imgui_widgets.cpp
  C:/Dev/imgui/imgui_demo.cpp
  C:/Dev/imgui/backends/imgui_impl_win32.cpp
  C:/Dev/imgui/backends/imgui_impl_dx12.cpp)
target_include_directories(imgui PUBLIC C:/Dev/imgui C:/Dev/imgui/backends)

add_executable(openntc-gui WIN32 openntc-gui/main.cpp
  openntc-gui/thread.cpp
  openntc-gui/model.cpp
  openntc-gui/rhi.cpp)
option(OPENNTC_COOP "Coop Vector Support" ON)
if(OPENNTC_COOP)
  set(AGILITY_DIR C:/Dev/microsoft.direct3d.d3d12.1.721.3-preview/build/native)
else()
  set(AGILITY_DIR C:/Dev/DirectXAgilitySDK/build/native)
endif()
target_include_directories(openntc-gui BEFORE PRIVATE ${AGILITY_DIR}/include)
target_compile_definitions(openntc-gui PRIVATE OPENNTC_COOP=$<BOOL:${OPENNTC_COOP}>)
add_custom_command(TARGET openntc-gui POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:openntc-gui>/D3D12
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AGILITY_DIR}/bin/x64/D3D12Core.dll        $<TARGET_FILE_DIR:openntc-gui>/D3D12
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AGILITY_DIR}/bin/x64/d3d12SDKLayers.dll  $<TARGET_FILE_DIR:openntc-gui>/D3D12)
target_link_directories(openntc-gui PRIVATE C:/Dev/DirectXTex/DirectXTex/Bin/Desktop_2022_Win10/x64/Debug)
target_link_libraries(openntc-gui PRIVATE D3D12.lib DXGI.lib D3DCompiler.lib DirectXTex.lib)
target_include_directories(openntc-gui PRIVATE C:/Dev/DirectXTex/)
target_include_directories(openntc-gui PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(openntc-gui PRIVATE imgui libopenntc)

# Preview DXC is required for cooperative vector support
set(DXC C:/Dev/dxc/bin/x64/dxc.exe)
set(DXC_PREVIEW C:/Dev/dxc_preview/bin/x64/dxc.exe)
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/openntc-gui)

set(SHADER_OUTPUTS "")
function(add_hlsl_variant COMPILER SRC_BASE OUT_BASE PROFILE_VS PROFILE_PS)
  foreach(stage vs ps)
    if(stage STREQUAL "vs")
      set(profile ${PROFILE_VS})
    else()
      set(profile ${PROFILE_PS})
    endif()
    set(out ${SHADER_DIR}/${OUT_BASE}_${stage}.cso)
    add_custom_command(
      OUTPUT ${out}
      COMMAND ${COMPILER} -T ${profile} -E ${stage}_main ${ARGN}
              "$<$<CONFIG:Debug,RelWithDebInfo>:-Zi;-Qembed_debug>"
              -Fo ${out} ${SHADER_DIR}/${SRC_BASE}.hlsl
      DEPENDS ${SHADER_DIR}/${SRC_BASE}.hlsl ${SHADER_DIR}/common.hlsli
      COMMENT "dxc (${profile}) ${SRC_BASE}.hlsl -> ${out}"
      COMMAND_EXPAND_LISTS
      VERBATIM)
    list(APPEND SHADER_OUTPUTS ${out})
  endforeach()
  set(SHADER_OUTPUTS ${SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()

function(add_hlsl COMPILER SRC_BASE PROFILE_VS PROFILE_PS)
  add_hlsl_variant(${COMPILER} ${SRC_BASE} ${SRC_BASE} ${PROFILE_VS} ${PROFILE_PS} ${ARGN})
  set(SHADER_OUTPUTS ${SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()

add_hlsl(${DXC} flat vs_6_6 ps_6_6 -D BPP_0_2)
add_hlsl(${DXC} pbr vs_6_6 ps_6_6 -D BPP_0_2)
add_hlsl(${DXC} cubemap vs_6_6 ps_6_6 -D BPP_0_2)
add_hlsl(${DXC} taa vs_6_6 ps_6_6 -D BPP_0_2)

add_hlsl(${DXC} pbr_ntc vs_6_6 ps_6_6 -D BPP_0_2)
add_hlsl_variant(${DXC} pbr_ntc pbr_ntc_bpp05 vs_6_6 ps_6_6 -D BPP_0_5)
if(OPENNTC_COOP)
  add_hlsl(${DXC_PREVIEW} pbr_ntc_coop vs_6_10 ps_6_10 -I C:/Dev/dxc_preview/inc/hlsl -D BPP_0_2)
  add_hlsl_variant(${DXC_PREVIEW} pbr_ntc_coop pbr_ntc_coop_bpp05 vs_6_10 ps_6_10 -I C:/Dev/dxc_preview/inc/hlsl -D BPP_0_5)
endif()

add_custom_target(shaders ALL DEPENDS ${SHADER_OUTPUTS})
add_dependencies(openntc-gui shaders)