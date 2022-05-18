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
#include "rtxParticles/common/programs/Camera.h"
#include "rtxParticles/common/programs/intersectSphere.h"
#include "rtxParticles/common/Particles.h"
#include <math_constants.h>

namespace pkd {
  namespace device {

    
    struct StackEntry {
      float t0,t1;
      unsigned int nodeID;
    };

#pragma region Util
    inline __device__ float mix(float a, float b, float t) {
        return a + t * (b - a);
    }

    inline __device__ float fract(float a) {
        return a - float(int(a));
    }

    inline __device__ float logx(float a, float base) {
        return log2f(a) / log2f(base);
    }
#pragma endregion

#pragma region Density

    // Transform vector to density volume space
    inline __device__ vec3f toDensityVolumeSpace(const AllPKDGeomData& self, const vec3f vec, const bool direction) {
        if (direction)
            return vec.x * self.densityContextBuffer[3] + vec.y * self.densityContextBuffer[4] + vec.z * self.densityContextBuffer[5];

        vec3f v = vec - self.densityContextBuffer[6];
        return v.x * self.densityContextBuffer[3] + v.y * self.densityContextBuffer[4] + v.z * self.densityContextBuffer[5];
    }

    // Get linearized voxel index
    inline __device__ int voxelIndex(const AllPKDGeomData& self, vec3i voxel) {
        vec3i voxelCount = vec3i(self.densityContextBuffer[2]);
        return voxelCount.y * voxelCount.z * max(min(voxel.x, voxelCount.x - 1), 0) + voxelCount.z * max(min(voxel.y, voxelCount.y - 1), 0) + max(min(voxel.z, voxelCount.z - 1), 0);
    }

    // Get bounds of voxel containing specified density volume space position
    inline __device__ box3f getVoxelBounds(const AllPKDGeomData& self, vec3f pos) {
        vec3f lowerBound = self.densityContextBuffer[0];
        vec3f upperBound = self.densityContextBuffer[1];
        vec3f boundSize = upperBound - lowerBound;
        vec3i voxelCount = vec3i(self.densityContextBuffer[2]);


        vec3f relPos = pos - lowerBound;
        relPos.x /= boundSize.x;
        relPos.y /= boundSize.y;
        relPos.z /= boundSize.z;
        vec3i voxelLower = vec3i(int(relPos.x * voxelCount.x), int(relPos.y * voxelCount.y), int(relPos.z * voxelCount.z));
        vec3i voxelUpper = vec3i(int(relPos.x * voxelCount.x) + 1, int(relPos.y * voxelCount.y) + 1, int(relPos.z * voxelCount.z) + 1);

        return box3f(vec3f((float(voxelLower.x) / float(voxelCount.x)) * boundSize.x,
            (float(voxelLower.y) / float(voxelCount.y)) * boundSize.y,
            (float(voxelLower.z) / float(voxelCount.z)) * boundSize.z) + lowerBound,
            vec3f((float(voxelUpper.x) / float(voxelCount.x)) * boundSize.x,
                (float(voxelUpper.y) / float(voxelCount.y)) * boundSize.y,
                (float(voxelUpper.z) / float(voxelCount.z)) * boundSize.z) + lowerBound);
    }

    // Get density at specified density volume space position (either with or without trilinear interpolation)
    inline __device__ float getDensity(const AllPKDGeomData& self, vec3f pos, bool interpolate) {

        vec3f lowerBound = self.densityContextBuffer[0];
        vec3f upperBound = self.densityContextBuffer[1];
        vec3f boundSize = upperBound - lowerBound;
        vec3i voxelCount = vec3i(self.densityContextBuffer[2]);

        vec3f relPos = pos - lowerBound;
        relPos.x /= boundSize.x;
        relPos.y /= boundSize.y;
        relPos.z /= boundSize.z;
        relPos.x *= voxelCount.x;
        relPos.y *= voxelCount.y;
        relPos.z *= voxelCount.z;

        if (!interpolate)
            return self.densityBuffer[voxelIndex(self, vec3i(int(relPos.x), int(relPos.y), int(relPos.z)))];
        
        vec3f h = vec3f(1.0f / voxelCount.x, 1.0f / voxelCount.y, 1.0f / voxelCount.z);

        relPos -= 0.5f * h;
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

    // Integrate density histogram along world space ray
    inline __device__ float integrateDensityHistogram(const AllPKDGeomData& self, const owl::Ray& ray,
        const float d_sample, const float d_cull, const float d_accum,
        float& B_d_min, float& B_d_sample, float& B_d_cull, float& B_d_accum, bool interpolate) {

        B_d_min = 0.0f;
        B_d_sample = 0.0f;
        B_d_cull = 0.0f;
        B_d_accum = 0.0f;

        owl::Ray transformedRay = owl::Ray(
            /* origin   : */ toDensityVolumeSpace(self, ray.origin, false),
            /* direction: */ toDensityVolumeSpace(self, ray.direction, true),
            /* tmin     : */ ray.tmin,
            /* tmax     : */ ray.tmax); // tmin and tmax can remain as no scaling, occurs

        float t0, t1;
        box3f bounds = box3f(self.densityContextBuffer[0], self.densityContextBuffer[1]);
        if (clipToBounds(transformedRay, bounds, t0, t1)) {
            float sum = 0.0f;

#pragma region Step size calculation
            float step = 0.1f;
            float absx = fabsf(transformedRay.direction.x);
            float absy = fabsf(transformedRay.direction.y);
            float absz = fabsf(transformedRay.direction.z);

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



            // March along ray
            for (float t = t0; t < t1; t += step) {

                float t_next = min(t + step, t1);

                float localDensity = getDensity(self, transformedRay.origin + (t + 0.5f * step) * transformedRay.direction, interpolate);

                // Approximate density histogram integral on [d_sample, d_max]
                if (t >= d_sample) {
                    B_d_sample += localDensity * ((t_next - t) / step);
                }
                else if (t_next >= d_sample) {
                    B_d_sample += ((t_next - d_sample) / step) * localDensity;
                }

                // Approximate density histogram integral on [d_cull, d_max]
                if (t >= d_cull) {
                    B_d_cull += localDensity * ((t_next - t) / step);
                }
                else if (t_next >= d_cull) {
                    B_d_cull += ((t_next - d_cull) / step) * localDensity;
                }

                // Approximate density histogram integral on [d_accum, d_max]
                if (t >= d_accum) {
                    B_d_accum += localDensity * ((t_next - t) / step);
                }
                else if (t_next >= d_accum) {
                    B_d_accum += ((t_next - d_accum) / step) * localDensity;
                }

                // Approximate density histogram integral on [d_min, d_max]
                B_d_min += localDensity * ((t_next - t) / step);

            }

            B_d_sample *= pixel_footprint * step;
            B_d_cull *= pixel_footprint * step;
            B_d_accum *= pixel_footprint * step;
            B_d_min *= pixel_footprint * step;
        }
        return B_d_min;
    }

#pragma endregion

#pragma region Confidence
    inline __device__ float accumulateConfidence(const float C, const float a, const float deltaEnk) {
        // Taken apart product from Eq. (8)
        float invC = 1.0f - C;
        invC *= pow(1.0f - a, deltaEnk);
        return 1.0f - invC;
    }

    inline __device__ float expectedUniqueParticles(const float N, const int k) {
        // Eq. (5)
        float M = 1.0f - (1.0f / N);
        return (1.0f - pow(M, k)) / (1.0f - M);
    }
#pragma endregion

#pragma region Acceptance Probability
    inline __device__ float normalCdf(const AllPKDGeomData& self, float x) {
        // Look up pre-computed Gaussian cumulative density function
        float n = self.normalCdfBuffer[1];
        float z_alpha = self.normalCdfBuffer[0];

        if (x < -z_alpha) return 0.0f;
        if (x > z_alpha) return 1.0f;

        return self.normalCdfBuffer[int(((x + z_alpha) / (2.0f * z_alpha)) * n) + 2];
    }

    inline __device__ int minUniqueParticles(const float B_d_accum, const float B_d_sample, const float C, const float a) {
        /* Equation (15) */
        float n = logx(1.0f - (B_d_accum / B_d_sample) * C, 1.0f - a);
        if (fract(n) > 0.0f) {
            return int(ceil(n));
        }
        return int(n) + 1;
    }

    inline __device__ int minSamples(const int n_ds, const float D_d_min_d_sample, const float M) {
        /* Equation (19) */
        return int(ceilf(logx(1.0f - (float(n_ds) / D_d_min_d_sample), M)));
    }

    inline __device__ float approximateBernoulliCdf(const AllPKDGeomData& self, const int N, const int k, const float p) {
        // Approximate Bernoulli distribution by Gaussian distribution
        float mean = float(N) * p;
        float var = float(N) * p * (1.0f - p);

        return 1.0f - normalCdf(self, (float(k) + 0.5f - mean) / sqrt(var));
    }

    inline __device__ float acceptanceProbability(const AllPKDGeomData& self, const float C_accum,
        const float B_d_accum, const float B_d_sample, const float B_d_min, const float a, const int N_budget) {

        int n_ds = minUniqueParticles(B_d_accum, B_d_sample, C_accum, a);


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


        return approximateBernoulliCdf(self, N_budget, k_ds, p_ds);
    }
#pragma endregion

#pragma region Accumulation Algorithm

    inline __device__ void accumAlgorithm(const AllPKDGeomData& self, const FrameState* fs, const int& pixelIdx, const owl::Ray& ray, const owl::Ray& centerRay, float& t, Random& rnd) {
        
        // Get sample depth
        float d_sample;
        if (fs->quant) {
            // quantised
            float t0, t1;
            clipToBounds(ray, getVoxelBounds(self, ray.origin + (t + 2.0f * self.particleRadius) * ray.direction), t0, t1);
            d_sample = t1;
        }
        else {
            // non-quantised
            d_sample = min(t + 2.0f * self.particleRadius, self.confidentDepthBufferPtr[pixelIdx]);
        }


        float d_cull = self.depthConfidenceCullBufferPtr[pixelIdx].x;
        float d_accum = self.depthConfidenceAccumBufferPtr[pixelIdx].x;

        // Sampled particle's footprint area
        float a_sample = min((CUDART_PI_F * self.particleRadius * self.particleRadius) / length(cross(fs->camera_screen_du, fs->camera_screen_dv)), 0.5f);

        if (d_cull == 1e20f || d_sample > d_cull) {  
            self.depthConfidenceCullBufferPtr[pixelIdx].x = d_sample;
            d_cull = d_sample;
        }
            
        // Density Histogram Integration
        float B_d_min, B_d_sample, B_d_cull, B_d_accum;

        B_d_min = integrateDensityHistogram(self, centerRay,
            d_sample, d_cull, d_accum,
            B_d_min, B_d_sample, B_d_cull, B_d_accum, fs->interp);

        // Look up acceptance probability
        float accProb = acceptanceProbability(self, self.depthConfidenceAccumBufferPtr[pixelIdx].y, B_d_accum, B_d_sample, B_d_min, a_sample, fs->nBudget);

        // ============================
        // == ACCUMULATION ALGORITHM ==
        // ============================

        // Accumulate Culling Confidence
        float Enk_cull = expectedUniqueParticles(max(1.0f, B_d_min - B_d_cull), self.depthConfidenceCullBufferPtr[pixelIdx].z);
        float deltaEnk_cull = Enk_cull - self.depthConfidenceCullBufferPtr[pixelIdx].w;

        self.depthConfidenceCullBufferPtr[pixelIdx].y = accumulateConfidence(self.depthConfidenceCullBufferPtr[pixelIdx].y, a_sample, deltaEnk_cull);
        self.depthConfidenceCullBufferPtr[pixelIdx].z += 1.0f;
        self.depthConfidenceCullBufferPtr[pixelIdx].w = Enk_cull;

        

        if (d_sample <= self.depthConfidenceAccumBufferPtr[pixelIdx].x) {
            //Update Accum Buffer
            float u = rnd();
            if (d_sample < d_accum
                && u <= accProb) {
                self.depthConfidenceAccumBufferPtr[pixelIdx].x = d_sample;
                self.depthConfidenceAccumBufferPtr[pixelIdx].y = a_sample;
                self.depthConfidenceAccumBufferPtr[pixelIdx].z = 1.0f;
                self.depthConfidenceAccumBufferPtr[pixelIdx].w = 0.0f;
            }
            else {
                // Accumulate Accum Confidence
                float Enk_accum = expectedUniqueParticles(max(1.0f, B_d_min - B_d_accum), self.depthConfidenceAccumBufferPtr[pixelIdx].z);
                float deltaEnk_accum = Enk_accum - self.depthConfidenceAccumBufferPtr[pixelIdx].w;

                self.depthConfidenceAccumBufferPtr[pixelIdx].y = accumulateConfidence(self.depthConfidenceAccumBufferPtr[pixelIdx].y, a_sample, deltaEnk_accum);
                self.depthConfidenceAccumBufferPtr[pixelIdx].z += 1.0f;
                self.depthConfidenceAccumBufferPtr[pixelIdx].w = Enk_accum;
            }
        }

        // Update cull buffer if accum buffer accumulated well enough
        if (B_d_accum * self.depthConfidenceAccumBufferPtr[pixelIdx].y > B_d_cull * self.depthConfidenceCullBufferPtr[pixelIdx].y) {

            self.depthConfidenceCullBufferPtr[pixelIdx].x = float(self.depthConfidenceAccumBufferPtr[pixelIdx].x);
            self.depthConfidenceCullBufferPtr[pixelIdx].y = float(self.depthConfidenceAccumBufferPtr[pixelIdx].y);
            self.depthConfidenceCullBufferPtr[pixelIdx].z = float(self.depthConfidenceAccumBufferPtr[pixelIdx].z);
            self.depthConfidenceCullBufferPtr[pixelIdx].w = float(self.depthConfidenceAccumBufferPtr[pixelIdx].w);
        }
    }

#pragma endregion
    

    /*! intersection program for 'allPKD' strategy - ie, all primitives
      are regular spheres ... */
    OPTIX_INTERSECT_PROGRAM(allPKD_intersect)()
    {
      const auto &self
        = owl::getProgramData<AllPKDGeomData>();

      const vec2i pixelID = owl::getLaunchIndex();
      const vec2i launchDim = owl::getLaunchDims();

      const int pixelIdx = pixelID.x + launchDim.x * pixelID.y;

      const FrameState* fs = &self.frameStateBuffer[0];
      
      owl::Ray ray(optixGetWorldRayOrigin(),
                   optixGetWorldRayDirection(),
                   optixGetRayTmin(),
                   optixGetRayTmax());

      Random rnd(pixelIdx,
          //0// for debugging
          fs->accumID//for real accumulation
      );

      owl::Ray centerRay = Camera::generateRay(*fs, float(pixelID.x) + .5f, float(pixelID.y) + .5f,
          rnd, 1e-6f, self.confidentDepthBufferPtr[pixelIdx]);

      bool converged = fs->samplesPerPixel * (fs->accumID - self.accumIDLastCulled[pixelIdx]) > fs->convergenceIterations;
      
      
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
            if (intersectSphere(particle, particleRadius, ray, tmp_hit_t)) {
                tmp_hit_primID = nodeID;

                if (!converged && // not yet converged
                    fs->probabilisticCulling && // culling enabled
                    fs->accumID > 0)
                    accumAlgorithm(self, fs, pixelIdx, ray, centerRay, tmp_hit_t, rnd);

            }
                    

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
