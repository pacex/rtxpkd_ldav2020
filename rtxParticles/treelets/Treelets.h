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
  
  struct PKDlet {
    //! bounding box of all particles (including the radius)
    box3f bounds;
    //! begin/end range in the common particles array
    size_t begin,end;
  };
    
  /*! a optix particle model that uses mini-pkd treelets, with an
      optix bvh buildt over those treelets */
  struct TreeletParticles : public OptixParticles
  {
    TreeletParticles();
    
    int maxTreeletSize = 1000;

    virtual void buildModel(Model::SP model, bool override_model = false) override;
    virtual void resizeFrameBuffer(const vec2i& newSize) override;

    // vec2i buffer that stores the begin/end for all treelets
    OWLBuffer treeletBuffer;
    OWLModule module;
    
    bool bruteForce = false;
  };
  
}
