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

#include "DRand48.h"
#include "owl/owl.h"

namespace pkd {
  namespace device {

    /*! the implicit state's ray we will intersect against */
    // rtBuffer<FrameState,1> frameStateBuffer;

    inline __device__ vec3f safe_normalize(const vec3f &v)
    {
      const vec3f vv = vec3f((fabsf(v.x) < 1e-8f) ? 1e-8f : v.x,
                             (fabsf(v.y) < 1e-8f) ? 1e-8f : v.y,
                             (fabsf(v.z) < 1e-8f) ? 1e-8f : v.z);
      return normalize(vv);
    }
    
    struct Camera {
      static __device__ owl::Ray generateRay(const FrameState &fs,
                                             float s, float t,
                                             Random &rnd, float tmin = 1e-6f /*Default TMIN*/, float tmax = 1e20f /*Default TMAX*/) 
      {
          if (!fs.orthoProjection) {

              /*
              * PERSPECTIVE
              */

              // const FrameState *fs = &frameStateBuffer[0];
                // const vec3f rd = 0.f; //camera_lens_radius * random_in_unit_disk(rnd);
                // const vec3f lens_offset = fs->camera_u * rd.x + fs->camera_v * rd.y;
              const vec3f origin = fs.camera_lens_center;// + lens_offset;
              const vec3f direction
                  = fs.camera_screen_00
                  + s * fs.camera_screen_du
                  + t * fs.camera_screen_dv
                  ;

              return owl::Ray(
                  // return optix::make_Ray(
                  /* origin   : */ origin,
                  /* direction: */ safe_normalize(direction),
                  /* tmin     : */ tmin,
                  /* tmax     : */ tmax);
        }

          /*
          * ORTHOGRAPHIC
          */

          // const FrameState *fs = &frameStateBuffer[0];
          // const vec3f rd = 0.f; //camera_lens_radius * random_in_unit_disk(rnd);
          // const vec3f lens_offset = fs->camera_u * rd.x + fs->camera_v * rd.y;
          const vec3f screen_00_to_center = fs.camera_screen_center - fs.camera_screen_00;
          const vec3f origin = fs.camera_lens_center + s * fs.camera_screen_du + t * fs.camera_screen_dv - screen_00_to_center;
          const vec3f direction = fs.camera_screen_center;

          return owl::Ray(
              // return optix::make_Ray(
              /* origin   : */ origin,
              /* direction: */ safe_normalize(direction),
              /* tmin     : */ tmin,
              /* tmax     : */ tmax);
        
      }
      // vec3f lens_center;
      // vec3f screen_00;
      // vec3f screen_du;
      // vec3f screen_dv;

    };
    
  }
}

