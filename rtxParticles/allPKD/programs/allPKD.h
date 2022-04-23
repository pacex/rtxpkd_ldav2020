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
#include "rtxParticles/treelets/Treelets.h"

namespace pkd {
  namespace device {

    struct AllPKDGeomData {
      Particle    *particleBuffer;
      float        particleRadius;
      int          particleCount;
      box3f        worldBounds;

      int* accumIDLastCulled;
      float4* depthConfidenceAccumBufferPtr;
      float4* depthConfidenceCullBufferPtr;
      float* confidentDepthBufferPtr;
      FrameState* frameStateBuffer;
      vec3f* densityContextBuffer;
      float* densityBuffer;
      float* normalCdfBuffer;
      uint32_t* normalBufferPtr;
    };

  }
}
