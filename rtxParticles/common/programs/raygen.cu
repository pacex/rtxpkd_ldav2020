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

      inline __device__ float integrateDensityHistogram(const RayGenData& self, const owl::Ray& ray,
          const float d_sample, const float d_cull, const float d_accum,
          float& B_d_min, float& B_d_sample, float& B_d_cull, float& B_d_accum) {

          B_d_min = 0.0f;
          B_d_sample = 0.0f;
          B_d_cull = 0.0f;
          B_d_accum = 0.0f;

          float t0, t1;
          box3f bounds = box3f(self.densityContextBuffer[0], self.densityContextBuffer[1]);
          if (clipToBounds(ray, bounds, t0, t1)) {
              float sum = 0.0f;

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

              bool past_d_sample = false;
              bool past_d_cull = false;
              bool past_d_accum = false;

              float t_last = t1;
              for (float t = t1 - 0.5 * step; t > t0; t -= step) {
                  
                  float localDensity = getDensity(self, ray.origin + t * ray.direction);


                  if (t > d_sample) {
                      B_d_sample += localDensity;
                  }
                  else if (!past_d_sample) {
                      B_d_sample *= step * pixel_footprint;
                      B_d_sample += getDensity(self, ray.origin + d_sample * ray.direction) * (t_last - d_sample) * pixel_footprint;
                      past_d_sample = true;
                  }

                  if (t > d_cull) {
                      B_d_cull += localDensity;
                  }
                  else if (!past_d_cull) {
                      B_d_cull *= step * pixel_footprint;
                      B_d_cull += getDensity(self, ray.origin + d_cull * ray.direction) * (t_last - d_cull) * pixel_footprint;
                      past_d_cull = true;
                  }

                  if (t > d_accum) {
                      B_d_accum += localDensity;
                  }
                  else if (!past_d_accum) {
                      B_d_accum *= step * pixel_footprint;
                      B_d_accum += getDensity(self, ray.origin + d_accum * ray.direction) * (t_last - d_accum) * pixel_footprint;
                      past_d_accum = true;
                  }

                  B_d_min += localDensity;

                  t_last = t + 0.5 * step;
              }

              B_d_min *= step * pixel_footprint;
              if (!past_d_sample)
                  B_d_sample = B_d_min;
              if (!past_d_cull)
                  B_d_cull = B_d_min;
              if (!past_d_accum)
                  B_d_accum = B_d_min;


              /*
              for (float t = t0 + 0.5f * step; t < t1; t += step) {
                  if (!past_d_sample && t > d_sample) {
                      d_sample_partCount = sum * step * pixel_footprint
                          + getDensity(self, ray.origin + d_sample * ray.direction) * (t - d_sample) * pixel_footprint;
                      past_d_sample = true;
                  }

                  if (!past_d_cull && t > d_cull) {
                      d_cull_partCount = sum * step * pixel_footprint
                          + getDensity(self, ray.origin + d_cull * ray.direction) * (t - d_cull) * pixel_footprint;
                      past_d_cull = true;
                  }

                  if (!past_d_accum && t > d_accum) {
                      d_accum_partCount = sum * step * pixel_footprint
                          + getDensity(self, ray.origin + d_accum * ray.direction) * (t - d_accum) * pixel_footprint;
                      past_d_accum = true;
                  }

                  sum += getDensity(self, ray.origin + t * ray.direction);
                  t_last = t;
              }
              sum *= step * pixel_footprint;
              sum += getDensity(self, ray.origin + t1 * ray.direction) * (t1 - t_last) * pixel_footprint;

              B_d_min = sum;
              B_d_sample = B_d_min - d_sample_partCount;
              B_d_cull = B_d_min - d_cull_partCount;
              B_d_accum = B_d_min - d_accum_partCount;
              */
          }
          return B_d_min;
#pragma region old    
          /*
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
          */
#pragma endregion
      }

#pragma endregion

#pragma region Confidence
      inline __device__ float accumulateConfidence(const float& N, const float& a, const int& k) {
          float M = 1.0f - (1.0f / N);
          float Enk = (1.0f - pow(M, k)) / (1.0f - M); //(5)

          return 1.0f - pow(1.0f - a, Enk); //(7)
      }

#pragma region Acceptance Probability
      inline __device__ float normalCdf(const RayGenData& self, float x) {
          float n = self.normalCdfBuffer[1];
          float z_alpha = self.normalCdfBuffer[0];

          if (x < -z_alpha) return 1.0f;
          if (x > z_alpha) return 0.0f;

          return self.normalCdfBuffer[int(((x + z_alpha) / (2.0f * z_alpha)) * n) + 2];
      }

      inline __device__ int minUniqueParticles(const float B_d_accum, const float B_d_sample, const float C, const float a) {
          /* Equation (15) */
          return max(1, int(ceilf(logx(1.0f - (B_d_accum / B_d_sample) * C, 1.0f - a))));
      }

      inline __device__ int minSamples(const int n_ds, const float D_d_min_d_sample, const float M) {
          /* Equation (19) */
          return int(ceilf(logx(1.0f - (float(n_ds) / D_d_min_d_sample), M)));
      }

      inline __device__ float approximateBernoulliCdf(const RayGenData& self, const int N, const int k, const float p) {
          float mean = float(N) * p;
          float var = float(N) * p * (1.0f - p);

          return normalCdf(self, (float(k) - 0.5f - mean) / var);
      }

      inline __device__ float acceptanceProbability(const RayGenData& self, const int& pixelIdx,
          const float B_d_accum, const float B_d_sample, const float B_d_min, const float& a) {
          
          int n_ds = minUniqueParticles(B_d_accum, B_d_sample, self.depthConfidenceAccumBufferPtr[pixelIdx].y, a);


          float D_d_min_d_sample = B_d_min - B_d_sample;

          /* Equation (20) */
          if (n_ds > 0.98f * D_d_min_d_sample) {
              return 0.0;
          }

          /* Equation (18) */
          float M = 1.0f - (1.0f / D_d_min_d_sample);

          /* Equation (19) */
          int k_ds = minSamples(n_ds, D_d_min_d_sample, M);

          /* Equation (22) */
          float p_ds = D_d_min_d_sample / B_d_min;

          int N_budget = 25;
          
          return approximateBernoulliCdf(self, N_budget, k_ds, p_ds);
      }
#pragma endregion
      /*
      inline __device__ float occludedParticles(const RayGenData& self, const owl::Ray& ray, const float& d, const float& C) {
          return integrateDensityHistogram(self, ray, d, 1e20f) * C; //(12)
      }
      */
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

      //DEBUG
#pragma region DEBUG
      if (pixelIdx == 0 && fs->accumID == 0) {/*
          float a = approximateBernoulliCdf(self, 100, 80, 0.5f);
          printf("%f\n", a);
          a = approximateBernoulliCdf(self, 100, 50, 0.45f);
          printf("%f\n", a);
          a = approximateBernoulliCdf(self, 20, 18, 0.4f);
          printf("%f\n", a);
          a = approximateBernoulliCdf(self, 32, 42, 0.9f);
          printf("%f\n", a);*/
      }
#pragma endregion

      //Culling by using depth as t_max
      if (fs->probabalisticCulling && self.depthConfidenceCullBufferPtr[pixelIdx].y >= fs->c_occ) {
          self.depthConfidenceCullBufferPtr[pixelIdx].w = self.depthConfidenceCullBufferPtr[pixelIdx].x;
      }

      owl::Ray centerRay = Camera::generateRay(*fs, float(pixelID.x) + .5f, float(pixelID.y) + .5f,
          rnd, 1e-6f, self.depthConfidenceCullBufferPtr[pixelIdx].w + 2.0f * self.radius);
      
      

      PerRayData prd;
      
      for (int s = 0; s < fs->samplesPerPixel; s++) {
        float u = float(pixelID.x + rnd());
        float v = float(pixelID.y + rnd());
        owl::Ray ray = Camera::generateRay(*fs, u, v, rnd, 1e-6f, self.depthConfidenceCullBufferPtr[pixelIdx].w + 2.0f * self.radius);
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

        //Depth Confidence Accumulation
        if (fs->probabalisticCulling && prd.particleID != -1 && fs->accumID > 0) {
            float d_sample = prd.t;
            float d_cull = self.depthConfidenceCullBufferPtr[pixelIdx].x;
            float d_accum = self.depthConfidenceAccumBufferPtr[pixelIdx].x;
            float a_sample = min((CUDART_PI_F * self.radius * self.radius) / length(cross(fs->camera_screen_du, fs->camera_screen_dv)), 0.5f); //Constant for now




            //Density Histogram Integration
            float B_d_min, B_d_sample, B_d_cull, B_d_accum;

            B_d_min = integrateDensityHistogram(self, ray,
                d_sample, d_cull, d_accum,
                B_d_min, B_d_sample, B_d_cull, B_d_accum);
            
            


            float N = max(1.0, B_d_min);

            float accProb = acceptanceProbability(self, pixelIdx, B_d_accum, B_d_sample, B_d_min, a_sample);

            if (fs->debugOutput && pixelIdx == 364383) {
                printf("\033[1;37m%i |\033[0;33m N= %f,\033[0;31m d_culled= %f,\033[0;36m d_sample= %f,\033[0;33m B_d_sample= %f,\033[0;36m d_cull= %f,\033[0;32m C_cull= %f,\033[0;33m B_d_cull= %f,\033[0;36m d_accum= %f,\033[0;32m C_accum= %f,\033[0;33m B_d_accum= %f,\033[0;34m accProb= %f\033[0m\n", 
                    fs->accumID, N, self.depthConfidenceCullBufferPtr[pixelIdx].w, d_sample, B_d_sample, d_cull, 
                    self.depthConfidenceCullBufferPtr[pixelIdx].y, B_d_cull, d_accum, self.depthConfidenceAccumBufferPtr[pixelIdx].y, B_d_accum, accProb);
            }

            if (d_sample <= d_cull) {
                self.depthConfidenceCullBufferPtr[pixelIdx].y = accumulateConfidence(N, a_sample, int(self.depthConfidenceCullBufferPtr[pixelIdx].z));
                self.depthConfidenceCullBufferPtr[pixelIdx].z += 1.0f;
            }

            if (d_sample <= self.depthConfidenceAccumBufferPtr[pixelIdx].x) {
                //Update Accum Buffer
                float u = rnd();
                if (d_sample <= d_accum
                    && u <= accProb) {
                    self.depthConfidenceAccumBufferPtr[pixelIdx].x = d_sample;
                    self.depthConfidenceAccumBufferPtr[pixelIdx].y = a_sample;
                    self.depthConfidenceAccumBufferPtr[pixelIdx].z = 1.0f;
                }
                else {
                    self.depthConfidenceAccumBufferPtr[pixelIdx].y = accumulateConfidence(N, a_sample, int(self.depthConfidenceAccumBufferPtr[pixelIdx].z));
                    self.depthConfidenceAccumBufferPtr[pixelIdx].z += 1.0f;
                }
            }

            if (B_d_accum * self.depthConfidenceAccumBufferPtr[pixelIdx].y > B_d_cull * self.depthConfidenceCullBufferPtr[pixelIdx].y) {
                
                self.depthConfidenceCullBufferPtr[pixelIdx].x = float(self.depthConfidenceAccumBufferPtr[pixelIdx].x);
                self.depthConfidenceCullBufferPtr[pixelIdx].y = float(self.depthConfidenceAccumBufferPtr[pixelIdx].y);
                self.depthConfidenceCullBufferPtr[pixelIdx].z = float(self.depthConfidenceAccumBufferPtr[pixelIdx].z);
            }
        }
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
          self.depthConfidenceAccumBufferPtr[pixelIdx] = vec3f(1e20f, 0.0, 1.0);
          self.depthConfidenceCullBufferPtr[pixelIdx] = vec4f(1e20f, 0.0, 1.0, 1e20f);
      }
        
      self.accumBufferPtr[pixelIdx] = col;
      self.normalAccumBufferPtr[pixelIdx] = norm;

      uint32_t rgba_col = make_rgba8(col / (fs->accumID + 1.f));
      uint32_t rgba_norm = make_rgba8(abs(norm) / (fs->accumID + 1.f));
      self.colorBufferPtr[pixelIdx] = rgba_col;

      //Debug
      //self.normalBufferPtr[pixelIdx] = rgba_norm;
      float a, b, c, d;
      self.normalBufferPtr[pixelIdx] = make_rgba8(vec4f(transferFunction(integrateDensityHistogram(self, centerRay, 0.0f, 0.0f, 0.0f, a, b, c, d)), 0.0f));
      
      vec3f lowerBound = self.densityContextBuffer[0];
      vec3f upperBound = self.densityContextBuffer[1];
      float boundSize = length(upperBound - lowerBound);

      self.depthBufferPtr[pixelIdx] = make_rgba8(vec4f(transferFunction(0.5f * self.depthConfidenceCullBufferPtr[pixelIdx].w / boundSize), 0.0f));
      self.coverageBufferPtr[pixelIdx] = make_rgba8(vec4f(transferFunction(self.depthConfidenceCullBufferPtr[pixelIdx].y), 0.0f));

    }
  }
}

