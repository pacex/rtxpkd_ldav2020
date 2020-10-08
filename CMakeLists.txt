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

project(RTX-Particles)

cmake_minimum_required(VERSION 2.8)

set(owl_dir ${CMAKE_CURRENT_SOURCE_DIR}/../owl)
# -------------------------------------------------------

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/common/cmake/")

if (WIN32)
    add_definitions(-DNOMINMAX)
    add_definitions(-DM_PI=3.14159265359f)
    add_definitions(-DOWL_HAVE_TBB=1)
    option(USE_MMPLD "Enable mmpld file type" OFF)
endif()

include(configure_owl)
message("OWL_INCLUDES ${OWL_INCLUDES}")
include_directories(${OWL_INCLUDES})
#include(configure_cuda)

# -------------------------------------------------------
if (WIN32)
set(CMAKE_CXX_FLAGS "/std:c++14 /arch:AVX2")
else()
set(CMAKE_CXX_FLAGS "-std=c++11 -mavx2")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (USE_MMPLD)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/submodules/mmpld_io)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/submodules/mmpld_io/include)
  ADD_DEFINITIONS(-DHAVE_MMPLD=1)
endif()

# -------------------------------------------------------

add_subdirectory(${owl_dir}
  external_owl
  EXCLUDE_FROM_ALL)
add_subdirectory(${owl_dir}/owl/common/viewerWidget
  external_owl_viewerWidget
  EXCLUDE_FROM_ALL)

# the actual back-ends:
add_subdirectory(rtxParticles)

