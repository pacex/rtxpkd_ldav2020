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

#include "rtxParticles/OptixParticles.h"
#include "rtxParticles/common/programs/raygen.h"
#include "rtxParticles/DensityVolume.h"
#include <chrono>
#include <numeric>

namespace pkd {

    extern "C" const char embedded_common_programs[];

    using device::RayGenData;

    int OptixParticles::rec_depth = 3;
    int OptixParticles::voxel_count = 64;

    OptixParticles::OptixParticles()
    {
        // create an optix context to render with, and do basic set-up
        context = owlContextCreate();
        module = owlModuleCreate(context, embedded_common_programs);
        
        std::cout << "todo: set stack size" << std::endl;
        
        frameStateBuffer = owlDeviceBufferCreate(context,
            OWL_USER_TYPE(device::FrameState),
            1, nullptr);

        // =======================================================
        // now, let's build up the world, innermost first
        // =======================================================

        // -------------------------------------------------------
        // set up miss prog 
        // -------------------------------------------------------
        OWLVarDecl missProgVars[] = {
          { /* sentinel to mark end of list */ }
        };
        // ........... create object  ............................
        PING; PRINT(module);
        OWLMissProg missProg
            = owlMissProgCreate(context, module, "miss_program", 0,//sizeof(MissProgData),
                missProgVars, -1);


        // -------------------------------------------------------
        // set up ray gen program
        // -------------------------------------------------------
        OWLVarDecl rayGenVars[] = {
          { "deviceIndex",     OWL_DEVICE, OWL_OFFSETOF(RayGenData,deviceIndex)},
          { "deviceCount",     OWL_INT,    OWL_OFFSETOF(RayGenData,deviceCount)},
          { "colorBuffer",     OWL_BUFPTR, OWL_OFFSETOF(RayGenData,colorBufferPtr)},
          { "accumBuffer",     OWL_BUFPTR, OWL_OFFSETOF(RayGenData,accumBufferPtr)},
          { "normalBuffer",    OWL_BUFPTR, OWL_OFFSETOF(RayGenData,normalBufferPtr)},
          { "normalAccumBuffer",    OWL_BUFPTR, OWL_OFFSETOF(RayGenData,normalAccumBufferPtr)},
          { "depthBuffer",      OWL_BUFPTR, OWL_OFFSETOF(RayGenData, depthBufferPtr)},
          { "depthConfidenceAccumBuffer",     OWL_BUFPTR, OWL_OFFSETOF(RayGenData, depthConfidenceAccumBufferPtr)},
          { "coverageBuffer",      OWL_BUFPTR, OWL_OFFSETOF(RayGenData, coverageBufferPtr)},
          { "depthConfidenceCullBuffer",     OWL_BUFPTR, OWL_OFFSETOF(RayGenData, depthConfidenceCullBufferPtr)},
          { "confidentDepthBuffer",     OWL_BUFPTR, OWL_OFFSETOF(RayGenData, confidentDepthBufferPtr)},
          { "particleBuffer",  OWL_BUFPTR, OWL_OFFSETOF(RayGenData,particleBuffer)},
          { "frameStateBuffer",OWL_BUFPTR, OWL_OFFSETOF(RayGenData,frameStateBuffer)},
          { "fbSize",          OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
          { "world",           OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
          { "rec_depth",       OWL_INT,    OWL_OFFSETOF(RayGenData,rec_depth)},
          { "radius",       OWL_FLOAT,    OWL_OFFSETOF(RayGenData,radius)},
          { "accumIDLastCulled",       OWL_BUFPTR,    OWL_OFFSETOF(RayGenData,accumIDLastCulled)},
          { /* sentinel to mark end of list */ }
        };

        // ........... create object  ............................
        this->rayGen
            = owlRayGenCreate(context, module, "raygen_program",
                sizeof(RayGenData),
                rayGenVars, -1);

        std::cout << "Setting rec depth: " << rec_depth << std::endl;
        owlRayGenSet1i(rayGen, "rec_depth", rec_depth);
        
        owlRayGenSetBuffer(rayGen, "frameStateBuffer", frameStateBuffer);

        //resizeFrameBuffer(vec2i(100));
    }

    void OptixParticles::setModel(Model::SP model, bool override_model)
    {
        model->particles.erase(std::unique(model->particles.begin(), model->particles.end(),
            [](auto const& lhs, auto const& rhs) { return lhs.pos == rhs.pos; }),
            model->particles.end());

        OptixParticles::model = model;

        std::cout << "building model: (#particles = " << model->particles.size() << ")" << std::endl;

        /* Density Field*/
        buildDensityField(vec3f(1.0f, 0.0f, 0.0f),
            vec3f(0.0f, 1.0f, 0.0f),
            vec3f(0.0f, 0.0f, 1.0f));

        /* Acceptance Probability */
        calculateNormalCdf();

        /* Model */
        buildModel(model, override_model);
        owlRayGenSetGroup(rayGen, "world", world);
        owlRayGenSetBuffer(rayGen, "particleBuffer", particleBuffer);
        owlRayGenSet1f(rayGen, "radius", model->radius);

        std::cout << "building programs" << std::endl;
        owlBuildPrograms(context);
        std::cout << "building pipeline" << std::endl;
        owlBuildPipeline(context);
        owlBuildSBT(context);
    }

    void OptixParticles::buildDensityField(vec3f xUnit, vec3f yUnit, vec3f zUnit) {
        DensityVolume::buildDensityField(OptixParticles::model, OptixParticles::voxel_count, xUnit, yUnit, zUnit);

        densityContextBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(DensityVolume::densityContext[0]), DensityVolume::densityContext.size(), DensityVolume::densityContext.data());
        densityBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(DensityVolume::particleDensity[0]), DensityVolume::particleDensity.size(), DensityVolume::particleDensity.data());
    }

    void OptixParticles::calculateNormalCdf() {

        // Build Gaussian lookup table for acceptance probability
        float z_alpha = 3.09f;
        int n = 256;

        std::cout << "building normalCDF lookup table: n = " << n << "; z_alpha = " << z_alpha << std::endl;

        float step = 2.0f * z_alpha / float(n);

        std::vector<float> cdf(n + 2);

        cdf[0] = z_alpha;
        cdf[1] = float(n);

        float x = -z_alpha;
        for (int i = 0; i < n; i++) {
            x += step;
            cdf[i + 2] = 0.5f * (1.0f + erff(x / M_SQRT2));
        }
        normalCdfBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(cdf[0]), cdf.size(), cdf.data());
    }


    void OptixParticles::render()
    {
        owlRayGenLaunch2D(rayGen, fbSize.x, fbSize.y);
    }

    void OptixParticles::resizeFrameBufferGeneral(const vec2i& newSize)
    {
        fbSize = newSize;

        //Convergence buffer
        if (!accumIDLastCulled)
            accumIDLastCulled = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);

        owlBufferResize(accumIDLastCulled, fbSize.x * fbSize.y);
        owlRayGenSetBuffer(rayGen, "accumIDLastCulled", accumIDLastCulled);

        //AccumBuffer
        if (!accumBuffer)
            accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, fbSize.x * fbSize.y, nullptr);
        
        owlBufferResize(accumBuffer, fbSize.x * fbSize.y);
        owlRayGenSetBuffer(rayGen, "accumBuffer", accumBuffer);
        
        //ColorBuffer
        if (!colorBuffer)
            colorBuffer = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);
        
        owlBufferResize(colorBuffer, fbSize.x * fbSize.y); 
        owlRayGenSetBuffer(rayGen, "colorBuffer", colorBuffer);

        //NormalAccumBuffer
        if (!normalAccumBuffer)
            normalAccumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, fbSize.x * fbSize.y, nullptr);

        owlBufferResize(normalAccumBuffer, fbSize.x * fbSize.y);
        owlRayGenSetBuffer(rayGen, "normalAccumBuffer", normalAccumBuffer);

        //NormalBuffer
        if (!normalBuffer)
            normalBuffer = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);

        owlBufferResize(normalBuffer, fbSize.x * fbSize.y);
        owlRayGenSetBuffer(rayGen, "normalBuffer", normalBuffer);

        //DepthConfidenceAccumBuffer
        if (!depthConfidenceAccumBuffer)
            depthConfidenceAccumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, fbSize.x * fbSize.y, nullptr);

        owlBufferResize(depthConfidenceAccumBuffer, fbSize.x * fbSize.y);
        owlRayGenSetBuffer(rayGen, "depthConfidenceAccumBuffer", depthConfidenceAccumBuffer);

        //DepthBuffer
        if (!depthBuffer)
            depthBuffer = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);

        owlBufferResize(depthBuffer, fbSize.x * fbSize.y);
        owlRayGenSetBuffer(rayGen, "depthBuffer", depthBuffer);

        //DepthConfidenceCullBuffer
        if (!depthConfidenceCullBuffer)
            depthConfidenceCullBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, fbSize.x * fbSize.y, nullptr);

        owlBufferResize(depthConfidenceCullBuffer, fbSize.x * fbSize.y);
        owlRayGenSetBuffer(rayGen, "depthConfidenceCullBuffer", depthConfidenceCullBuffer);

        //ConfidentDepthBuffer
        if (!confidentDepthBuffer)
            confidentDepthBuffer = owlDeviceBufferCreate(context, OWL_FLOAT, fbSize.x * fbSize.y, nullptr);

        owlBufferResize(confidentDepthBuffer, fbSize.x * fbSize.y);
        owlRayGenSetBuffer(rayGen, "confidentDepthBuffer", confidentDepthBuffer);
        

        //CoverageBuffer
        if (!coverageBuffer)
            coverageBuffer = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);

        owlBufferResize(coverageBuffer, fbSize.x * fbSize.y);
        owlRayGenSetBuffer(rayGen, "coverageBuffer", coverageBuffer);

        owlRayGenSet1i(rayGen, "deviceCount", owlGetDeviceCount(context));
        owlRayGenSet2i(rayGen, "fbSize", fbSize.x, fbSize.y);
        
    }

    void OptixParticles::updateFrameState(device::FrameState& fs)
    {
        owlBufferUpload(frameStateBuffer, &fs);
    }

    uint32_t* OptixParticles::mapColorBuffer()
    {
        if (!colorBuffer) return nullptr;
        return (uint32_t*)owlBufferGetPointer(colorBuffer, 0);
    }

    void OptixParticles::unmapColorBuffer()
    {
        assert(colorBuffer);
    }

    uint32_t* OptixParticles::mapNormalBuffer()
    {
        if (!normalBuffer) return nullptr;
        return (uint32_t*)owlBufferGetPointer(normalBuffer, 0);
    }

    void OptixParticles::unmapNormalBuffer()
    {
        assert(normalBuffer);
    }

    uint32_t* OptixParticles::mapDepthBuffer()
    {
        if (!depthBuffer) return nullptr;
        return (uint32_t*)owlBufferGetPointer(depthBuffer, 0);
    }

    void OptixParticles::unmapDepthBuffer()
    {
        assert(depthBuffer);
    }

    uint32_t* OptixParticles::mapCoverageBuffer()
    {
        if (!coverageBuffer) return nullptr;
        return (uint32_t*)owlBufferGetPointer(coverageBuffer, 0);
    }

    void OptixParticles::unmapCoverageBuffer()
    {
        assert(coverageBuffer);
    }
}
