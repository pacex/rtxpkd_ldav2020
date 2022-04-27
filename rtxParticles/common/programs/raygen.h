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

#pragma once

#include "rtxParticles/common/Particles.h"

namespace pkd {
  namespace device {

    struct RayGenData {
      int deviceIndex;
      int deviceCount;
      OptixTraversableHandle world;
      int       rec_depth;
      int*      accumIDLastCulled;
      float     radius;
      vec2ui    fbSize;
      uint32_t *colorBufferPtr;
      uint32_t *normalBufferPtr;
      uint32_t* depthBufferPtr;
      uint32_t* coverageBufferPtr;
      float4   *accumBufferPtr;
      float4   *normalAccumBufferPtr;
      float4* depthConfidenceAccumBufferPtr; //(d,C,k,Enk)
      float4* depthConfidenceCullBufferPtr; //(d,C,k,Enk)
      float* confidentDepthBufferPtr;
      Particle *particleBuffer;
      FrameState *frameStateBuffer;
    };
    
  }
}
