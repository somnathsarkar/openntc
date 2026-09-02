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

set(OPENNTC_IMGUI_DIR "" CACHE PATH "Dear ImGui source directory")
if(OPENNTC_IMGUI_DIR STREQUAL "")
  message(FATAL_ERROR "OPENNTC_IMGUI_DIR should be source path to a Dear ImGui repo")
endif()
add_library(imgui STATIC
  ${OPENNTC_IMGUI_DIR}/imgui.cpp
  ${OPENNTC_IMGUI_DIR}/imgui_draw.cpp
  ${OPENNTC_IMGUI_DIR}/imgui_tables.cpp
  ${OPENNTC_IMGUI_DIR}/imgui_widgets.cpp
  ${OPENNTC_IMGUI_DIR}/imgui_demo.cpp
  ${OPENNTC_IMGUI_DIR}/backends/imgui_impl_win32.cpp
  ${OPENNTC_IMGUI_DIR}/backends/imgui_impl_dx12.cpp)
target_include_directories(imgui PUBLIC ${OPENNTC_IMGUI_DIR} ${OPENNTC_IMGUI_DIR}/backends)

set(OPENNTC_DIRECTXTEX_INCLUDE_DIR "" CACHE PATH "Directory containing DirectXTex.h")
set(OPENNTC_DIRECTXTEX_LIB_DIR "" CACHE PATH "Directory containing Debug and/or Release directories with DirectXTex.lib")
if(OPENNTC_DIRECTXTEX_INCLUDE_DIR STREQUAL "" OR OPENNTC_DIRECTXTEX_LIB_DIR STREQUAL "")
  message(FATAL_ERROR "Set OPENNTC_DIRECTXTEX_INCLUDE_DIR and OPENNTC_DIRECTXTEX_LIB_DIR prior to building")
endif()
set(DIRECTXTEX_LIB ${OPENNTC_DIRECTXTEX_LIB_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>/DirectXTex.lib)

add_executable(openntc-gui WIN32 openntc-gui/main.cpp
  openntc-gui/thread.cpp
  openntc-gui/model.cpp
  openntc-gui/rhi.cpp
  openntc-gui/datapack.cpp)
# Cooperative-vector support needs preview Agility SDK and preview dxc
option(OPENNTC_COOP "Coop Vector Support" OFF)
set(OPENNTC_AGILITY_DIR "" CACHE PATH "Agility SDK 'build/native' dir (Preview SDK when OPENNTC_COOP is ON)")
if(OPENNTC_AGILITY_DIR STREQUAL "")
  message(FATAL_ERROR "Set OPENNTC_AGILITY_DIR to the Agility SDK build/native directory")
endif()
set(AGILITY_DIR ${OPENNTC_AGILITY_DIR})
target_include_directories(openntc-gui BEFORE PRIVATE ${AGILITY_DIR}/include)
target_compile_definitions(openntc-gui PRIVATE OPENNTC_COOP=$<BOOL:${OPENNTC_COOP}>)
add_custom_command(TARGET openntc-gui POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:openntc-gui>/D3D12
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AGILITY_DIR}/bin/x64/D3D12Core.dll $<TARGET_FILE_DIR:openntc-gui>/D3D12
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AGILITY_DIR}/bin/x64/d3d12SDKLayers.dll $<TARGET_FILE_DIR:openntc-gui>/D3D12)
target_link_libraries(openntc-gui PRIVATE D3D12.lib DXGI.lib D3DCompiler.lib ${DIRECTXTEX_LIB})
target_include_directories(openntc-gui PRIVATE ${OPENNTC_DIRECTXTEX_INCLUDE_DIR})
target_include_directories(openntc-gui PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(openntc-gui PRIVATE imgui libopenntc)

# Shader compiler: Any dxc with SM 6.6 support
# Coop vector support: Preview dxc with SM 6.10 support
find_program(OPENNTC_DXC dxc DOC "dxc.exe with Shader Model 6.6 support")
if(NOT OPENNTC_DXC)
  message(FATAL_ERROR "dxc not found. Add to PATH or set OPENNTC_DXC")
endif()
set(DXC ${OPENNTC_DXC})
if(OPENNTC_COOP)
  set(OPENNTC_DXC_PREVIEW "" CACHE FILEPATH "Preview dxc.exe (SM 6.10)")
  set(OPENNTC_DXC_PREVIEW_INCLUDE "" CACHE PATH "Preview dxc include directory")
  if(OPENNTC_DXC_PREVIEW STREQUAL "" OR OPENNTC_DXC_PREVIEW_INCLUDE STREQUAL "")
    message(FATAL_ERROR "OPENNTC_COOP=ON requires OPENNTC_DXC_PREVIEW and OPENNTC_DXC_PREVIEW_INCLUDE")
  endif()
  set(DXC_PREVIEW ${OPENNTC_DXC_PREVIEW})
endif()
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/openntc-gui)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADER_OUTPUTS "")
function(add_hlsl_variant COMPILER SRC_BASE OUT_BASE PROFILE_VS PROFILE_PS)
  foreach(stage vs ps)
    if(stage STREQUAL "vs")
      set(profile ${PROFILE_VS})
    else()
      set(profile ${PROFILE_PS})
    endif()
    set(out ${SHADER_OUT_DIR}/${OUT_BASE}_${stage}.cso)
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
add_hlsl_variant(${DXC} pbr_ntc pbr_ntc_bpp10 vs_6_6 ps_6_6 -D BPP_1_0)
if(OPENNTC_COOP)
  add_hlsl(${DXC_PREVIEW} pbr_ntc_coop vs_6_10 ps_6_10 -I ${OPENNTC_DXC_PREVIEW_INCLUDE} -D BPP_0_2)
  add_hlsl_variant(${DXC_PREVIEW} pbr_ntc_coop pbr_ntc_coop_bpp05 vs_6_10 ps_6_10 -I ${OPENNTC_DXC_PREVIEW_INCLUDE} -D BPP_0_5)
  add_hlsl_variant(${DXC_PREVIEW} pbr_ntc_coop pbr_ntc_coop_bpp10 vs_6_10 ps_6_10 -I ${OPENNTC_DXC_PREVIEW_INCLUDE} -D BPP_1_0)
endif()

add_custom_target(shaders ALL DEPENDS ${SHADER_OUTPUTS})
add_dependencies(openntc-gui shaders)

set(OPENNTC_ASSET_FILES
  ${CMAKE_SOURCE_DIR}/data/specular_cube.dds
  ${CMAKE_SOURCE_DIR}/data/dfg.dds
  ${CMAKE_SOURCE_DIR}/data/sh.txt
  ${CMAKE_SOURCE_DIR}/data/knob.bin)
foreach(f ${OPENNTC_ASSET_FILES})
  if(NOT EXISTS ${f})
    message(FATAL_ERROR "Missing binary asset: ${f}")
  endif()
endforeach()

add_executable(pack_data tools/pack_data.cpp)

set(PACK_INPUTS "")
foreach(cso ${SHADER_OUTPUTS})
  get_filename_component(blob_name ${cso} NAME_WE)
  list(APPEND PACK_INPUTS "${blob_name}=${cso}")
endforeach()
list(APPEND PACK_INPUTS
  "specular_cube.dds=${CMAKE_SOURCE_DIR}/data/specular_cube.dds"
  "dfg.dds=${CMAKE_SOURCE_DIR}/data/dfg.dds"
  "sh.txt=${CMAKE_SOURCE_DIR}/data/sh.txt"
  "knob.bin=${CMAKE_SOURCE_DIR}/data/knob.bin")

set(PACK_FILE_DEPS "")
foreach(pair ${PACK_INPUTS})
  string(REGEX REPLACE "^[^=]+=" "" dep ${pair})
  list(APPEND PACK_FILE_DEPS ${dep})
endforeach()

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/data.bin
  COMMAND pack_data ${CMAKE_BINARY_DIR}/data.bin ${PACK_INPUTS}
  DEPENDS pack_data ${PACK_FILE_DEPS}
  COMMENT "pack_data -> data.bin"
  VERBATIM)
add_custom_target(datapack ALL DEPENDS ${CMAKE_BINARY_DIR}/data.bin)
add_dependencies(openntc-gui datapack)
add_custom_command(TARGET openntc-gui POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/data.bin $<TARGET_FILE_DIR:openntc-gui>/data.bin)