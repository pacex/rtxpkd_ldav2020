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

#pragma once

#include "rtxParticles/OptixParticles.h"

namespace pkd {

  /*! a optix particle model that doesn't use any pkd whatsoever, and
      simply represents each particle with a optix sphere primimive,
      with an optix bhv built over all those prims */
  struct NoPKDParticles : public OptixParticles
  {
    NoPKDParticles();
    virtual void buildModel(Model::SP model, bool override_model = false) override;
    void resizeFrameBuffer(const vec2i& newSize) override;
  };

}
