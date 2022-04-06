#pragma once

// gdt
#include "owl/owl.h"
#include "owl/common/math/LinearSpace.h"
// ours
#include "rtxParticles/common/Particles.h"
#include "rtxParticles/common/programs/FrameState.h"

#include <chrono>
#include <queue>

namespace pkd {
	static class DensityVolume {
    public:
        static std::vector<vec3f> densityContext;
        static std::vector<float> particleDensity;

    private:
        static float gauss(float x, float mu, float sigma);
        static box3i getBoundingVoxels(const vec3f p, const float radius, const box3f bounds, const vec3i voxelCount);
        static float getOverlap(const vec3f p, const float radius, const vec3i voxel, const box3f bounds, const vec3i voxelCount, const float integratedGauss[], const int nGauss);
        static vec3f getVoxelLower(const vec3i voxel, const box3f bounds, const vec3i voxelCount);
        static vec3f getVoxelUpper(const vec3i voxel, const box3f bounds, const vec3i voxelCount);
        static vec3f toDensityVolumeSpace(const vec3f vec, const vec3f x, const vec3f y, const vec3f z, const vec3f c, const bool direction);

    public:
        static void buildDensityField(Model::SP model, const int n, const vec3f xUnit, const vec3f yUnit, const vec3f zUnit);
	};
}