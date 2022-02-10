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

#pragma region Density

      inline __device__ int voxelIndex(const RayGenData& self, vec3i voxel) {
          int n = int(self.densityContextBuffer[2].x);
          return n * n * max(min(voxel.x, n-1), 0) + n * max(min(voxel.y, n-1), 0) + max(min(voxel.z, n-1), 0);
      }

      inline __device__ float getDensity(const RayGenData& self, vec3f pos) {

          vec3f lowerBound = self.densityContextBuffer[0];
          vec3f upperBound = self.densityContextBuffer[1];
          vec3f boundSize = upperBound - lowerBound;
          int n = int(self.densityContextBuffer[2].x);
          float h = 1.0f / self.densityContextBuffer[2].x;


          vec3f relPos = pos - lowerBound;
          relPos.x /= boundSize.x;
          relPos.y /= boundSize.y;
          relPos.z /= boundSize.z;
          relPos *= n;

          relPos -= vec3f(0.5f * h, 0.5f * h, 0.5f * h);
          relPos.x = fmaxf(0.0f, relPos.x);
          relPos.y = fmaxf(0.0f, relPos.y);
          relPos.z = fmaxf(0.0f, relPos.z);

          //Trilinear Interpolation
          float xInterp00 = mix(self.densityBuffer[voxelIndex(self, vec3i(int(relPos.x), int(relPos.y), int(relPos.z)))],
              self.densityBuffer[voxelIndex(self, vec3i(int(relPos.x) + 1, int(relPos.y), int(relPos.z)))], fract(relPos.x));
          float xInterp01 = mix(self.densityBuffer[voxelIndex(self, vec3i(int(relPos.x), int(relPos.y), int(relPos.z) + 1))],
              self.densityBuffer[voxelIndex(self, vec3i(int(relPos.x) + 1, int(relPos.y), int(relPos.z) + 1))], fract(relPos.x));
          float xInterp10 = mix(self.densityBuffer[voxelIndex(self, vec3i(int(relPos.x), int(relPos.y) + 1, int(relPos.z)))],
              self.densityBuffer[voxelIndex(self, vec3i(int(relPos.x) + 1, int(relPos.y) + 1, int(relPos.z)))], fract(relPos.x));
          float xInterp11 = mix(self.densityBuffer[voxelIndex(self, vec3i(int(relPos.x), int(relPos.y) + 1, int(relPos.z) + 1))],
              self.densityBuffer[voxelIndex(self, vec3i(int(relPos.x) + 1, int(relPos.y) + 1, int(relPos.z) + 1))], fract(relPos.x));

          float yxInterp0 = mix(xInterp00, xInterp10, fract(relPos.y));
          float yxInterp1 = mix(xInterp01, xInterp11, fract(relPos.y));

          return mix(yxInterp0, yxInterp1, fract(relPos.z));
          
      }

      inline __device__ float integrateDensityHistogram(const RayGenData& self, const owl::Ray& ray, const float& d_min, const float& d_max, float step_relative = 0.1f) {

          

          box3f bounds = box3f(self.densityContextBuffer[0], self.densityContextBuffer[1]);

          float sum = 0.0f;
          
          float t0, t1;
          if (clipToBounds(ray, bounds, t0, t1)) {

              t0 = max(t0, d_min);
              t1 = min(t1, d_max);

              const FrameState* fs = &self.frameStateBuffer[0];

              float step = step_relative * length(bounds.upper - bounds.lower);
              float pixel_footprint = length(cross(fs->camera_screen_du, fs->camera_screen_dv));

              float t_last = t0;
              for (float t = t0; t < t1; t += step) {
                  sum += getDensity(self, ray.origin + t * ray.direction);
                  t_last = t;
              }
              sum *= step * pixel_footprint;

              sum += getDensity(self, ray.origin + t1 * ray.direction) * (t1 - t_last) * pixel_footprint;
          }
          return sum;
      }

      inline __device__ float densityHistogramDepth(const RayGenData& self, const owl::Ray& ray, float particleCount) {



          box3f bounds = box3f(self.densityContextBuffer[0], self.densityContextBuffer[1]);

          float sum = 0.0f;

          float t0, t1;
          if (clipToBounds(ray, bounds, t0, t1)) {

#pragma region Step size calculation
              float step = 0.1f;
              float absx = fabsf(ray.direction.x);
              float absy = fabsf(ray.direction.y);
              float absz = fabsf(ray.direction.z);

              if (absx > absy && absx > absz) {
                  step = ((bounds.upper.x - bounds.lower.x) / self.densityContextBuffer[2].x) / absx;
              }
              else if (absy > absx && absy > absz) {
                  step = ((bounds.upper.y - bounds.lower.y) / self.densityContextBuffer[2].y) / absy;
              }
              else if (absz > absx && absz > absx) {
                  step = ((bounds.upper.z - bounds.lower.z) / self.densityContextBuffer[2].z) / absz;
              }
#pragma endregion

              const FrameState* fs = &self.frameStateBuffer[0];
              float pixel_footprint = length(cross(fs->camera_screen_du, fs->camera_screen_dv));

              float t_last = t0;
              for (float t = t0; t < t1; t += step) {

                  float localDensity = getDensity(self, ray.origin + (t + 0.5f * step) * ray.direction);
                  sum += localDensity;

                  if (sum > particleCount) {
                      return t + (1.0f - (sum - particleCount) / localDensity) * step;
                  }

                  t_last = t;
              }
          }
          return t1;
      }

#pragma endregion

#pragma region Confidence
      inline __device__ float accumulateConfidence(const float& N, const float& a, const int& k) {
          float M = 1.0f - (1.0f / N);
          float Enk = (1.0f - pow(M, k)) / (1.0f - M); //(5)

          return 1.0f - pow(1.0f - a, Enk); //(7)
      }

      inline __device__ float normalCdf(const RayGenData& self, float x) {
          float n = self.normalCdfBuffer[1];
          float z_alpha = self.normalCdfBuffer[0];

          if (x < -z_alpha) return 0.0f;
          if (x > z_alpha) return 1.0f;

          return self.normalCdfBuffer[int(((x + z_alpha) / (2.0f * z_alpha)) * n) + 2];
      }

      inline __device__ float acceptanceProbability(const RayGenData& self, const int& pixelIdx, const float& B_d, const float& B_ds, const float& B_dmin, const float& a) {
          
          /* Equation (15) */
          float n_ds = logx(1.0f - (B_d / B_ds) * self.depthConfidenceCullBufferPtr[pixelIdx].y, 1.0f - a);


          float B_dmin_ds = B_dmin - B_ds;

          /* Equation (20) */
          if (n_ds > 0.98f * B_dmin_ds) {
              return 0.0;
          }

          /* Equation (18) */
          float M = 1.0f - (1.0f / B_dmin_ds);

          /* Equation (19) */
          int k_ds = int(ceilf(logx(1.0f - (n_ds / B_ds), M)));

          /* Equation (22) */
          float p_ds = B_dmin_ds / (B_dmin_ds + B_ds);
          float q_ds = 1.0f - p_ds;

          int N_budget = 50;
          /* Equation (21) */
          float mean = float(N_budget) * p_ds;
          float var = float(N_budget) * p_ds * q_ds;

          
          return 1.0f - normalCdf(self, (k_ds + 0.5f - mean) / var);
      }

      inline __device__ float occludedParticles(const RayGenData& self, const owl::Ray& ray, const float& d, const float& C) {
          return integrateDensityHistogram(self, ray, d, 1e20f) * C; //(12)
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

      
      
      //Depth Culling
      
      float confidentDepth = 1e20f;
      /*
      float N = max(1.0f, self.depthConfidenceCullBufferPtr[pixelIdx].z);
      float M = 1.0f - (1.0f / N);
      float Enk = (1.0f - pow(M, self.depthConfidenceCullBufferPtr[pixelIdx].y)) / (1.0f - M);
      */

      if (fs->probabalisticCulling) {
          confidentDepth = self.depthConfidenceCullBufferPtr[pixelIdx].x + self.radius;
      }

      PerRayData prd;
      
      for (int s = 0; s < fs->samplesPerPixel; s++) {
        float u = float(pixelID.x + rnd());
        float v = float(pixelID.y + rnd());
        owl::Ray ray = Camera::generateRay(*fs, u, v, rnd, 1e-6f, confidentDepth);
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

        /*
        //Depth Confidence Accumulation
        if (fs->probabalisticCulling && prd.particleID != -1 && fs->accumID > 0) {
            float s_depth = length(self.particleBuffer[prd.particleID].pos - ray.origin);
            
            //Update depth if sampled depth is larger
            if (self.depthConfidenceCullBufferPtr[pixelIdx].y < 1.0f)
                self.depthConfidenceCullBufferPtr[pixelIdx].x = s_depth;
            else
                self.depthConfidenceCullBufferPtr[pixelIdx].x = max(s_depth, self.depthConfidenceCullBufferPtr[pixelIdx].x);
            
            //Increment k
            self.depthConfidenceCullBufferPtr[pixelIdx].y += 1.0f;         
        }*/
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
      else {
          //Framestate changed -> reset buffers, restart accumulation
          self.depthConfidenceAccumBufferPtr[pixelIdx] = vec3f(1e20f, 0.0, 0.0);

          self.depthConfidenceCullBufferPtr[pixelIdx] = vec3f(1e20f, 0.0f, 0.0f);

          if (fs->probabalisticCulling) {
              float s_a = min((CUDART_PI_F * self.radius * self.radius) / length(cross(fs->camera_screen_du, fs->camera_screen_dv)), 0.5f); //Constant for now
              float minUniqueParts = logx(1.0f - fs->c_occ, 1.0f - s_a);

              owl::Ray centerRay = Camera::generateRay(*fs, float(pixelID.x) + .5f, float(pixelID.y) + .5f, rnd, 1e-6f, 1e20f);
              float d = densityHistogramDepth(self, centerRay, minUniqueParts);
              self.depthConfidenceCullBufferPtr[pixelIdx] = vec3f(d, 0.0f, 0.0f);
          }
      }
        
      self.accumBufferPtr[pixelIdx] = col;
      self.normalAccumBufferPtr[pixelIdx] = norm;

      uint32_t rgba_col = make_rgba8(col / (fs->accumID + 1.f));
      uint32_t rgba_norm = make_rgba8(abs(norm) / (fs->accumID + 1.f));
      self.colorBufferPtr[pixelIdx] = rgba_col;

      //Debug
      //self.normalBufferPtr[pixelIdx] = rgba_norm;
      self.normalBufferPtr[pixelIdx] = make_rgba8(vec4f(transferFunction(self.depthConfidenceCullBufferPtr[pixelIdx].z), 0.0f));
      
      vec3f lowerBound = self.densityContextBuffer[0];
      vec3f upperBound = self.densityContextBuffer[1];
      float boundSize = length(upperBound - lowerBound);

      self.depthBufferPtr[pixelIdx] = make_rgba8(vec4f(transferFunction(0.5f * self.depthConfidenceCullBufferPtr[pixelIdx].x / boundSize), 0.0f));
      //self.coverageBufferPtr[pixelIdx] = make_rgba8(vec4f(transferFunction(self.depthConfidenceCullBufferPtr[pixelIdx].y / minUniqueParts), 0.0f));

    }
  }
}

