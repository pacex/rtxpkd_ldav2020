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
#include "owl/owl.h"
#include "owl/common/math/LinearSpace.h"
// ours
#include "rtxParticles/common/Particles.h"
#include "rtxParticles/common/programs/FrameState.h"
#include "DensityVolume.h"

#include <chrono>
#include <queue>

namespace pkd {

  /*! the root object for our particles, containing the model, the
      setup code for otpix, and the 'api' to interact with the
      viewer. TODO: split off the optix setup code, and do optix setup
      in three derived classes, one each for the three strategies
      (noPKD,allPKD,and hybrid) */
  struct OptixParticles {
    OptixParticles();

    void setModel(Model::SP model, bool override_model = false);

    virtual void buildModel(Model::SP model, bool override_model = false) = 0;

    void buildDensityField(vec3f xUnit, vec3f yUnit, vec3f zUnit);

    void calculateNormalCdf();
    
    virtual void resizeFrameBuffer(const vec2i& newSize) = 0;
    void resizeFrameBufferGeneral(const vec2i& newSize);

    void updateFrameState(device::FrameState &fs);

    uint32_t *mapColorBuffer();
    void unmapColorBuffer();

    uint32_t* mapNormalBuffer();
    void unmapNormalBuffer();

    uint32_t* mapDepthBuffer();
    void unmapDepthBuffer();

    uint32_t* mapCoverageBuffer();
    void unmapCoverageBuffer();

    void render();
    
    /*! size of current frame buffer */
    vec2i fbSize { -1,-1 };

    OWLContext context = 0;
    OWLModule module = 0;
    OWLBuffer frameStateBuffer = 0;
    OWLBuffer colorBuffer = 0;
    OWLBuffer normalBuffer = 0;
    OWLBuffer depthBuffer = 0;
    OWLBuffer coverageBuffer = 0;
    OWLBuffer accumBuffer = 0;
    OWLBuffer normalAccumBuffer = 0;
    OWLBuffer depthConfidenceAccumBuffer = 0;
    OWLBuffer depthConfidenceCullBuffer = 0;
    OWLBuffer confidentDepthBuffer = 0;
    OWLGroup  world = 0;
    OWLRayGen rayGen = 0;
    OWLBuffer particleBuffer = 0;
    OWLBuffer densityBuffer = 0;
    OWLBuffer densityContextBuffer = 0;
    OWLBuffer normalCdfBuffer = 0;
    OWLBuffer accumIDLastCulled = 0;

    static int rec_depth;
    static int voxel_count;
    static float radius;

    Model::SP model;

    DensityVolume densityVolume;
  };

}
