// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
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

#include "DRand48.h"

namespace pkd {
  namespace device {

    
    __device__ float schlick(float cosine, float ref_idx) {
      float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
      r0 = r0 * r0;
      return r0 + (1.0f - r0)*pow((1.0f - cosine), 5.0f);
    }

    __device__ bool refract(const vec3f& v, const vec3f& n, float ni_over_nt, vec3f& refracted) {
      vec3f uv = normalize(v);
      float dt = dot(uv, n);
      float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
      if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
      }
      else
        return false;
    }

    inline __device__ vec3f reflect(const vec3f &v, const vec3f &n)
    {
      return v - 2.0f*dot(v, n)*n;
    }


    inline __device__ vec3f random_in_unit_disk(Random &local_rand_state) {
      vec3f p;
      do {
        p = 2.0f*vec3f(local_rand_state(), local_rand_state(), 0) - vec3f(1, 1, 0);
      } while (dot(p, p) >= 1.0f);
      return p;
    }


#define RANDVEC3F vec3f(rnd(),rnd(),rnd())

    inline __device__ vec3f random_in_unit_sphere(Random &rnd) {
      vec3f p;
      do {
        p = 2.0f*RANDVEC3F - vec3f(1, 1, 1);
      } while (dot(p,p) >= 1.0f);
      return p;
    }
    
  }
}

