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

// #include <optix_world.h>
#include "noPKD.h"
#include "rtxParticles/common/programs/PerRayData.h"
#include "rtxParticles/common/programs/common.h"
#include "rtxParticles/common/programs/intersectSphere.h"
#include "rtxParticles/common/Particles.h"

namespace pkd {
  namespace device {
   
    /*! intersection program for 'noPKD' strategy - ie, all primitives
        are regular spheres ... */
    OPTIX_INTERSECT_PROGRAM(noPKD_intersect)()
    {
      const int primID = optixGetPrimitiveIndex();

      const auto &self
        = owl::getProgramData<NoPKDGeomData>();
      
      owl::Ray ray(optixGetWorldRayOrigin(),
                   optixGetWorldRayDirection(),
                   optixGetRayTmin(),
                   optixGetRayTmax());

      const pkd::Particle particle = self.particleBuffer[primID];
      float tmp_hit_t = ray.tmax;
      if (intersectSphere(particle,self.particleRadius,ray,tmp_hit_t)) {
        optixReportIntersection(tmp_hit_t,0);
      }
    }


    OPTIX_BOUNDS_PROGRAM(noPKD_bounds)(const void  *geomData,
                                         box3f       &primBounds,
                                         const int    primID)
    {
      const NoPKDGeomData &self = *(const NoPKDGeomData*)geomData;

      const pkd::Particle &particle = self.particleBuffer[primID];

      primBounds.lower = particle.pos - self.particleRadius;
      primBounds.upper = particle.pos + self.particleRadius;     
    }
    
    OPTIX_CLOSEST_HIT_PROGRAM(noPKD_closest_hit)()
    {
      const int primID = optixGetPrimitiveIndex();
      PerRayData &prd = owl::getPRD<PerRayData>();
      prd.particleID  = primID;
      prd.t           = optixGetRayTmax();
    }
    
  }  
}
