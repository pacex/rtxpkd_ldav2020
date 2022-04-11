# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

# ======================================================================== #
# Modified 2019-2020 VISUS - University of Stuttgart                       #
# ======================================================================== #

find_package(OpenGL REQUIRED)
add_subdirectory(glad)

cuda_compile_and_embed(embedded_common_programs common/programs/raygen.cu)
cuda_compile_and_embed(embedded_noPKD_programs  noPKD/programs/noPKD.cu)
cuda_compile_and_embed(embedded_allPKD_programs allPKD/programs/allPKD.cu)
cuda_compile_and_embed(embedded_treelets_programs treelets/programs/treelets.cu)

set(sources
  viewer.cpp
  noPKD/NoPKD.cpp
  allPKD/AllPKD.cpp
  treelets/Treelets.cpp
  OptixParticles.cpp
  DensityVolume.cpp
  Util.cpp

  loaders/loaders.cpp
  loaders/p4.cpp
  loaders/XYZ.cpp
  loaders/Random.cpp
  loaders/Grid.cpp
  loaders/PKD_Loader.cpp
  )
if (USE_MMPLD)
  set(sources ${sources} loaders/MMPLD_Loader.cpp)
endif()

add_executable(rtxParticleViewer
  ${sources}
  ${embedded_common_programs}
  ${embedded_noPKD_programs}
  ${embedded_allPKD_programs}
  ${embedded_treelets_programs}
  )

if(WIN32)
  include_directories(${GLUT_INCLUDE_DIR})
  find_package(TBB REQUIRED CONFIG)
  target_link_libraries(rtxParticleViewer
    owl_viewerWidget_static
    glad
    ${OPENGL_LIBRARIES}
    ${TBB_IMPORTED_TARGETS}
  )
  install(TARGETS rtxParticleViewer RUNTIME DESTINATION "bin")
  # https://stackoverflow.com/questions/30049180/strip-filename-shortest-extension-by-cmake-get-filename-removing-the-last-ext
  string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${GLUT_glut_LIBRARY_RELEASE})
  set(freeglut_dll "${CMAKE_MATCH_1}.dll")
  install(FILES "${freeglut_dll}" DESTINATION "bin")
  get_target_property(tbb_dll_release TBB::tbb IMPORTED_LOCATION_RELEASE)
  get_target_property(tbb_dll_debug TBB::tbb IMPORTED_LOCATION_DEBUG)
  get_target_property(tbbmalloc_dll_release TBB::tbbmalloc IMPORTED_LOCATION_RELEASE)
  get_target_property(tbbmalloc_dll_debug TBB::tbbmalloc IMPORTED_LOCATION_DEBUG)
  get_target_property(tbbmalloc_proxy_dll_release TBB::tbbmalloc_proxy IMPORTED_LOCATION_RELEASE)
  get_target_property(tbbmalloc_proxy_dll_debug TBB::tbbmalloc_proxy IMPORTED_LOCATION_DEBUG)
  install(FILES ${tbb_dll_release} DESTINATION "bin")
  install(FILES ${tbb_dll_debug} DESTINATION "bin")
  install(FILES ${tbbmalloc_dll_release} DESTINATION "bin")
  install(FILES ${tbbmalloc_dll_debug} DESTINATION "bin")
  install(FILES ${tbbmalloc_proxy_dll_release} DESTINATION "bin")
  install(FILES ${tbbmalloc_proxy_dll_debug} DESTINATION "bin")
else()
  target_link_libraries(rtxParticleViewer
    owl_viewerWidget
${OWL_NG_LIBRARIES}
    tbb
  )
endif()
