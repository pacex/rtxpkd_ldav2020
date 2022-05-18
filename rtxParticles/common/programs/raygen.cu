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


#include "rtxParticles/common/programs/PerRayData.h"
#include "rtxParticles/common/programs/Camera.h"
#include "rtxParticles/common/programs/sampling.h"
#include "rtxParticles/common/Particles.h"
#include "rtxParticles/common/programs/raygen.h"
#include "rtxParticles/common/programs/intersectSphere.h"

#include <thrust/device_vector.h>
#include <math_constants.h>


// #define COLOR_CODING 1

namespace pkd {
  namespace device {
    
#pragma region Util
      inline __device__
          int32_t make_8bit(const float f)
      {
          return min(255, max(0, int(f * 256.f)));
      }

      inline __device__ float mix(float a, float b, float t) {
          return a + t * (b - a);
      }

      inline __device__ float fract(float a) {
          return a - float(int(a));
      }

      inline __device__ float logx(float a, float base) {
          return log2f(a) / log2f(base);
      }

      inline __device__ vec3f transferFunction(const float f)
      {
          const int NUM_POINTS = 7;
          const vec3f colors[NUM_POINTS + 1] = {
            vec3f(0),
            vec3f(0,0,1),
            vec3f(0,1,1),
            vec3f(0,1,0),
            vec3f(1,1,0),
            vec3f(1,0,0),
            vec3f(1,1,1)
          };
          if (f <= 0.f) return vec3f(0.f);
          float f_scaled = f * (NUM_POINTS - 1);
          int segment = int(f_scaled);
          if (segment >= (NUM_POINTS - 1)) return vec3f(1.f);
          return colors[segment] + fmodf(f_scaled, 1.f) * (colors[segment + 1] - colors[segment]);
      }

      inline __device__
          int32_t make_rgba8(const vec4f color)
      {
          return
              (make_8bit(color.x) << 0) +
              (make_8bit(color.y) << 8) +
              (make_8bit(color.z) << 16);
      }
#pragma endregion

#pragma region TraceRay
      inline __device__ vec3f traceRay(const RayGenData& self,
          owl::Ray& ray, Random& rnd, PerRayData& prd)
      {
          const bool lastFieldOfParticleIsScalarValue = false;

          vec3f attenuation = 1.f;
          vec3f ambientLight(.8f);

          /* iterative version of recursion, up to depth 50 */
          for (int depth = 0;true;depth++) {
              prd.particleID = -1;

              owl::traceRay(/*accel to trace against*/self.world,
                  /*the ray to trace*/ ray,
                  // /*numRayTypes*/1,
                  /*prd*/prd
                  ,
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT
              );
              if (prd.particleID == -1) {
                  // miss...
                  return attenuation * ambientLight;
              }

              const Particle particle = self.particleBuffer[prd.particleID];
              vec3f N = (ray.origin + prd.t * ray.direction) - particle.pos;
              // printf("normal %f %f %f\n",N.x,N.y,N.z);
              if (dot(N, (vec3f)ray.direction) > 0.f)
                  N = -N;
              N = normalize(N);

              // hardcoded albedo for now:
#if COLOR_CODING
              const vec3f albedo
                  = randomColor(prd.treeletID);
#else
        //const vec3f albedo
        //  = //prd.primID == 0 ? vec3f(.1,.6,.3) :
        //  (lastFieldOfParticleIsScalarValue)
        //  ? transferFunction(.1f*sqrtf(particle.fieldValue))
        //  : randomColor(1+particle.matID);
              const vec3f albedo
                  = randomColor(0);
#endif
              // hard-coded for the 'no path tracing' case:
              if (self.rec_depth == 0)
                  return albedo * (.2f + .6f * fabsf(dot(N, (vec3f)ray.direction)));


              attenuation *= albedo;

              if (depth >= self.rec_depth) {
                  // ambient term:
                  return 0.1f;
              }

              const vec3f scattered_origin = ray.origin + prd.t * ray.direction;
              const vec3f scattered_direction = N + random_in_unit_sphere(rnd);
              ray = owl::Ray(/* origin   : */ scattered_origin,
                  /* direction: */ safe_normalize(scattered_direction),
                  /* tmin     : */ 1e-3f,
                  /* tmax     : */ 1e+8f);
          }
      }
#pragma endregion


    OPTIX_MISS_PROGRAM(miss_program)()
    // RT_PROGRAM void miss_program()
    {
      /*! nothing to do - we initialize prd before trace */
    }

    /*! the actual ray generation program - note this has no formal
      function parameters, but gets its paramters throught the 'pixelID'
      and 'pixelBuffer' variables/buffers declared above */
    OPTIX_RAYGEN_PROGRAM(raygen_program)()
    {
      const RayGenData &self = owl::getProgramData<RayGenData>();
      const vec2i pixelID = owl::getLaunchIndex();
      const vec2i launchDim = owl::getLaunchDims();
  
      if (pixelID.x >= self.fbSize.x) return;
      if (pixelID.y >= self.fbSize.y) return;
      const int pixelIdx = pixelID.x+self.fbSize.x*pixelID.y;
      const int debugPixelIdx = int(0.5f * self.fbSize.x + 0.44f * self.fbSize.y * self.fbSize.x);

      // for multi-gpu: only render every deviceCount'th column of 32 pixels:
      if (((pixelID.x/32) % self.deviceCount) != self.deviceIndex)
        return;
      
      uint64_t clock_begin = clock64();
      const FrameState *fs = &self.frameStateBuffer[0];
      int pixel_index = pixelID.y * launchDim.x + pixelID.x;

      //Accumulate color and normal across multiple samples
      vec4f col(0.f);
      vec4f norm(0.f,0.f,0.f,1.f);

      Random rnd(pixel_index,
          //0// for debugging
          fs->accumID//for real accumulation
      );


      // Initialize Depth Confidence Buffers
      if (fs->accumID <= 0){
          //Framestate changed -> reset buffers, restart accumulation
          self.depthConfidenceAccumBufferPtr[pixelIdx] = vec4f(1e20f,   // Depth
              0.0f,     // Confidence
              1.0f,     // sample count k
              0.0f);    // expected number of unique particles E(n(k))

          self.depthConfidenceCullBufferPtr[pixelIdx] = vec4f(1e20f,    // Depth
              0.0f,     // Confidence
              1.0f,     // sample count k
              0.0f);    // expected number of unique particles E(n(k))

          self.confidentDepthBufferPtr[pixelIdx] = 1e20f;   // t_max buffer, future rays will be cast with this t_max
          self.accumIDLastCulled[pixelIdx] = 0;             // accumID at which last culling decision occurred
      }

      // Has no culling decision occurred for long enough? -> If so, assume convergence
      bool converged = fs->samplesPerPixel * (fs->accumID - self.accumIDLastCulled[pixelIdx]) > fs->convergenceIterations;
      
      const int KERNEL_SIZE = fs->kernelSize;

      // If not converged, check if we can cull
      if (fs->probabilisticCulling && !converged) {
          float d_repr = -1.0f;
          float c_repr = 0.0f;
          int pixel_count = 0;

          // Traverse neighborhood kernel
          for (int i = max(0,pixelID.x - KERNEL_SIZE); i <= min(self.fbSize.x - 1, pixelID.x + KERNEL_SIZE); i++) {
              for (int j = max(0, pixelID.y - KERNEL_SIZE); j <= min(self.fbSize.y - 1, pixelID.y + KERNEL_SIZE); j++) {
                  int pxID = i + self.fbSize.x * j;

                  d_repr = max(d_repr, self.depthConfidenceCullBufferPtr[pxID].x);
                  c_repr += self.depthConfidenceCullBufferPtr[pxID].y;

                  pixel_count++;
              }
          }

          c_repr /= float(pixel_count);

          // If culling possible, set t_max and update accumID buffer
          if (c_repr >= fs->c_occ && self.confidentDepthBufferPtr[pixelIdx] > d_repr) {
              self.confidentDepthBufferPtr[pixelIdx] = d_repr;
              self.accumIDLastCulled[pixelIdx] = fs->accumID;
          }
      }

      PerRayData prd;
      
      for (int s = 0; s < fs->samplesPerPixel; s++) {
        float u = float(pixelID.x + rnd());
        float v = float(pixelID.y + rnd());
        owl::Ray ray = Camera::generateRay(*fs, u, v, rnd, 1e-6f, self.confidentDepthBufferPtr[pixelIdx]);
        col += vec4f(traceRay(self, ray, rnd,prd),1);

        //Normals
        vec3f Normal(0.f);
        if (prd.particleID != -1) {
            const Particle particle = self.particleBuffer[prd.particleID];
            Normal = vec3f((ray.origin + prd.t * ray.direction) - particle.pos);
            if (dot(Normal, (vec3f)ray.direction) > 0.f)
                Normal = -Normal;
            Normal = normalize(Normal);
        }
        norm += vec4f(Normal, 0.f);
      }

      //Accumulate color and normal across multiple samples
      col = col / float(fs->samplesPerPixel);
      norm = norm / float(fs->samplesPerPixel);
      
      

      uint64_t clock_end = clock64();
      if (fs->heatMapEnabled) {
        float t = (clock_end-clock_begin)*fs->heatMapScale;
        if (t >= 256.f*256.f*256.f)
          col = vec4f(1,0,0,1);
        else {
          uint64_t ti = uint64_t(t);
          col.x = ((ti >> 16) & 255)/255.f;
          col.y = ((ti >> 8) & 255)/255.f;
          col.z = ((ti >> 0) & 255)/255.f;
        }
      }
    
      if (fs->accumID > 0) {
        col = col + (vec4f)self.accumBufferPtr[pixelIdx];
        norm = norm + (vec4f)self.normalAccumBufferPtr[pixelIdx];
      }
        
      self.accumBufferPtr[pixelIdx] = col;
      self.normalAccumBufferPtr[pixelIdx] = norm;

      if (fs->probabilisticCulling && fs->debugOutput && pixelIdx == debugPixelIdx) {
          vec4f c = col / (fs->accumID + 1.f);
          printf("(%f)\n", (c.x + c.y + c.z) / 3.0f);
      }

      uint32_t rgba_col = make_rgba8(col / (fs->accumID + 1.f));
      uint32_t rgba_norm = make_rgba8(abs(norm) / (fs->accumID + 1.f));
      

      //Debug
      if (fs->debugOutput && pixelIdx == debugPixelIdx) {
          rgba_col = make_rgba8(vec4f(1.0f, 0.0f, 1.0f, 1.0f));
      }


      self.normalBufferPtr[pixelIdx] = rgba_norm;

      // Write convergence status to normal buffer
      if (fs->probabilisticCulling) {
          if (converged) {
              //Converged
              self.normalBufferPtr[pixelIdx] = make_rgba8(vec4f(0.0f, 1.0f, 0.0f, 1.0f));
          }
          else {
              //Not Converged
              self.normalBufferPtr[pixelIdx] = make_rgba8(vec4f(1.0f, 0.0f, 0.0f, 1.0f));
          }
      } 

      self.colorBufferPtr[pixelIdx] = rgba_col;

      self.depthBufferPtr[pixelIdx] = make_rgba8(vec4f(transferFunction(self.confidentDepthBufferPtr[pixelIdx] / 1e20f), 0.0f));
      self.coverageBufferPtr[pixelIdx] = make_rgba8(vec4f(transferFunction(self.depthConfidenceCullBufferPtr[pixelIdx].y), 0.0f));

    }
  }
}

