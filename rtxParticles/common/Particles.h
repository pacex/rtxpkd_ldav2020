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

// gdt
#include "owl/llowl.h"
#include "owl/common/math/box.h"
// std
#include <vector>
#include <memory>
#include <random>

#ifdef HAVE_MMPLD
#include "mmpld.h"
#endif

namespace pkd {
  using namespace owl;
  using namespace owl::common;

  struct Particle {
    vec3f pos;
    float dim;
  };
    
  /*! the entire set of tubes, including all links - everything we
    wnat to render */
  struct Model {
    typedef std::shared_ptr<Model> SP;

    inline __both__ box3f getBounds(int primID) const
    {
      const vec3f pos = particles[primID].pos;
      return box3f()
        .including(pos - radius)
        .including(pos + radius);
    }
    
    box3f getBounds() const
    {
      box3f bounds;
      for (int i=0;i<particles.size();++i)
        bounds.extend(getBounds(i));
      return bounds;
    }
    
    /*! radius, common to all particles */
    float radius;
    
    std::vector<Particle> particles;
  };
}
