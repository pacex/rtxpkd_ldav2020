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

#include "rtxParticles/allPKD/programs/allPKD.h"
#include "rtxParticles/common/programs/PerRayData.h"
#include "rtxParticles/common/programs/intersectSphere.h"
#include "rtxParticles/common/Particles.h"

namespace pkd {
  namespace device {

    
    struct StackEntry {
      float t0,t1;
      unsigned int nodeID;
    };
    

    /*! intersection program for 'allPKD' strategy - ie, all primitives
      are regular spheres ... */
    OPTIX_INTERSECT_PROGRAM(allPKD_intersect)()
    {
      const auto &self
        = owl::getProgramData<AllPKDGeomData>();
      
      owl::Ray ray(optixGetWorldRayOrigin(),
                   optixGetWorldRayDirection(),
                   optixGetRayTmin(),
                   optixGetRayTmax());
      
      float t0, t1;
      if (!clipToBounds(ray,self.worldBounds,t0,t1))
        return;
      
      int nodeID = 0;
      float tmp_hit_t = t1;
      int tmp_hit_primID = -1;

      enum { STACK_DEPTH = 32 };
      StackEntry stackBase[STACK_DEPTH];
      StackEntry *stackPtr = stackBase;
      
      const int dir_sign[3] = {
        ray.direction.x < 0.f,
        ray.direction.y < 0.f,
        ray.direction.z < 0.f
      };
      const float org[3] = {
        ray.origin.x,
        ray.origin.y,
        ray.origin.z
      };
      const float rdir[3] = {
        (fabsf(ray.direction.x) <= 1e-8f) ? 1e8f : 1.f/ray.direction.x,
        (fabsf(ray.direction.y) <= 1e-8f) ? 1e8f : 1.f/ray.direction.y,
        (fabsf(ray.direction.z) <= 1e-8f) ? 1e8f : 1.f/ray.direction.z,
      };
      unsigned int const numParticles = self.particleCount;
      float const particleRadius = self.particleRadius;

      while (1) {
        // while we have anything to traverse ...
        
        while (1) {
          // while we can go down
          const pkd::Particle particle = self.particleBuffer[nodeID];
          int const dim = particle.dim;

          const float t_slab_lo = (particle.pos[dim] - particleRadius - org[dim]) * rdir[dim];
          const float t_slab_hi = (particle.pos[dim] + particleRadius - org[dim]) * rdir[dim];

          const float t_slab_nr = fminf(t_slab_lo,t_slab_hi);
          const float t_slab_fr = fmaxf(t_slab_lo,t_slab_hi);

          // -------------------------------------------------------
          // compute potential sphere interval, and intersect if necessary
          // -------------------------------------------------------
          const float sphere_t0 = fmaxf(t0,t_slab_nr);
          const float sphere_t1 = fminf(fminf(t_slab_fr,t1),tmp_hit_t);

          if (sphere_t0 < sphere_t1) {
            if (intersectSphere(particle,particleRadius, ray, tmp_hit_t))
              tmp_hit_primID = nodeID;
          }
        
          // -------------------------------------------------------
          // compute near and far side intervals
          // -------------------------------------------------------
          const float nearSide_t0 = t0;
          const float nearSide_t1 = fminf(fminf(t_slab_fr, t1), tmp_hit_t);
                                          
          const float farSide_t0 = fmaxf(t0, t_slab_nr);
          const float farSide_t1 = fminf(t1,tmp_hit_t);


          // -------------------------------------------------------
          // logic 
          // -------------------------------------------------------
          const int  nearSide_nodeID = 2*nodeID+1+dir_sign[dim];
          const int  farSide_nodeID  = 2*nodeID+2-dir_sign[dim];
          
          const bool nearSide_valid = nearSide_nodeID < numParticles;
          const bool farSide_valid = farSide_nodeID < numParticles;

          const bool need_nearSide = nearSide_valid && nearSide_t0 < nearSide_t1;
          const bool need_farSide  = farSide_valid  && farSide_t0  < farSide_t1;

          if (!(need_nearSide || need_farSide)) break; // pop ...

          if (need_nearSide && need_farSide) {
              stackPtr->t0 = farSide_t0;
              stackPtr->t1 = farSide_t1;
              stackPtr->nodeID = farSide_nodeID;
              ++stackPtr;

              nodeID = nearSide_nodeID;
              t0 = nearSide_t0;
              t1 = nearSide_t1;
              continue;
          }

          nodeID = need_nearSide ? nearSide_nodeID : farSide_nodeID;
          t0 = need_nearSide ? nearSide_t0 : farSide_t0;
          t1 = need_nearSide ? nearSide_t1 : farSide_t1;

        }
        // -------------------------------------------------------
        // pop
        // -------------------------------------------------------
        while (1) {
          if (stackPtr == stackBase) {
            // can't pop any more - done.
            if (tmp_hit_primID >= 0 && tmp_hit_t < ray.tmax) {
              optixReportIntersection(tmp_hit_t,0,tmp_hit_primID);
            }
            return;
          }
          --stackPtr;
          t0 = stackPtr->t0;
          t1 = min(stackPtr->t1, tmp_hit_t);
          nodeID = stackPtr->nodeID;
          if (t1 <= t0)
            continue;
          break;
        }
      }
    }

    OPTIX_BOUNDS_PROGRAM(allPKD_bounds)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
    {
      auto &self = *(const AllPKDGeomData*)geomData;
      primBounds = self.worldBounds;
    }
    
    
    OPTIX_CLOSEST_HIT_PROGRAM(allPKD_closest_hit)()
    {
      PerRayData &prd = owl::getPRD<PerRayData>();
      prd.particleID  = optixGetAttribute_0();
      prd.t           = optixGetRayTmax();
    }
    
  } // ::pkd::device
} // ::pkd
