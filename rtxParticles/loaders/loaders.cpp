// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// ======================================================================== //
// Modified 2019-2020 VISUS - University of Stuttgart                       //
// ======================================================================== //

// gdt
#include "rtxParticles/common/Particles.h"

#include "rtxParticles/loaders/MMPLD_Loader.h"
#include "rtxParticles/loaders/Random.h"
#include "rtxParticles/loaders/XYZ.h"
#include "rtxParticles/loaders/PKD_Loader.h"

namespace pkd {

  Model::SP loadP4(const std::string &fileName);

  Model::SP loadGrid(const std::string &fileName);
  
  Model::SP loadParticleModel(const std::string &fileName)
  {
    auto ext_start = fileName.find_last_of('.');
    auto ext = fileName.substr(ext_start);

#if HAVE_MMPLD
    if (ext == ".mmpld")
      return loadMMPLD(fileName);
#endif
    if (ext == ".grid")
      return loadGrid(fileName);
    if (ext == ".random")
      return loadRandom(fileName);
    if (ext == ".xyz")
      return loadXYZ(fileName);
    if (ext == ".p4")
      return loadP4(fileName);
    if (ext == ".pkd")
      return loadPKD(fileName);

    throw std::runtime_error("un-supported file format '"+ext+"', or loader not compiled in");
  }
  
}
