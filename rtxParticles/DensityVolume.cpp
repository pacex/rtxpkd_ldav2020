#include "rtxParticles/OptixParticles.h"
#include "rtxParticles/common/programs/raygen.h"
#include "DensityVolume.h"
#include <chrono>
#include <numeric>
#include <iostream>


namespace pkd {

    std::vector<vec3f> DensityVolume::densityContext = std::vector<vec3f>();
    std::vector<float> DensityVolume::particleDensity = std::vector<float>();

#pragma region Splatting

    /* Get all voxels a particle intersects */
    box3i DensityVolume::getBoundingVoxels(const vec3f p, const float radius, const box3f bounds, const vec3i voxelCount) {
        vec3f relPosLower = (p - vec3f(radius)) - bounds.lower;
        vec3f relPosUpper = (p + vec3f(radius)) - bounds.lower;
        vec3f boundsSize = bounds.upper - bounds.lower;
        relPosLower.x /= boundsSize.x; relPosLower.y /= boundsSize.y; relPosLower.z /= boundsSize.z;
        relPosUpper.x /= boundsSize.x; relPosUpper.y /= boundsSize.y; relPosUpper.z /= boundsSize.z;

        vec3i voxelLower = vec3i(max(0, min(voxelCount.x - 1, int(relPosLower.x * voxelCount.x))),
            max(0, min(voxelCount.y - 1, int(relPosLower.y * voxelCount.y))),
            max(0, min(voxelCount.z - 1, int(relPosLower.z * voxelCount.z))));

        vec3i voxelUpper = vec3i(max(0, min(voxelCount.x - 1, int(relPosUpper.x * voxelCount.x))),
            max(0, min(voxelCount.y - 1, int(relPosUpper.y * voxelCount.y))),
            max(0, min(voxelCount.z - 1, int(relPosUpper.z * voxelCount.z))));

        return box3i(voxelLower, voxelUpper);
    }

    float DensityVolume::gauss(float x, float mu, float sigma) {
        return (0.398942f * pow(2.71828f, -(0.5f * (x - mu) * (x - mu)) / (sigma * sigma))) / sigma;
    }

    /* Get lower bounds of voxel */
    vec3f DensityVolume::getVoxelLower(const vec3i voxel, const box3f bounds, const vec3i voxelCount) {
        vec3f boundsSize = bounds.upper - bounds.lower;
        return bounds.lower + vec3f((float(voxel.x) / float(voxelCount.x)) * boundsSize.x,
            (float(voxel.y) / float(voxelCount.y)) * boundsSize.y,
            (float(voxel.z) / float(voxelCount.z)) * boundsSize.z);
    }

    /* Get upper bounds of voxel */
    vec3f DensityVolume::getVoxelUpper(const vec3i voxel, const box3f bounds, const vec3i voxelCount) {
        vec3f boundsSize = bounds.upper - bounds.lower;
        return bounds.lower + vec3f((float(voxel.x + 1) / float(voxelCount.x)) * boundsSize.x,
            (float(voxel.y + 1) / float(voxelCount.y)) * boundsSize.y,
            (float(voxel.z + 1) / float(voxelCount.z)) * boundsSize.z);
    }

    /* How much volume of the particle is also inside the given voxel? */
    float DensityVolume::getOverlap(const vec3f p, const float radius, const vec3i voxel, const box3f bounds, const vec3i voxelCount, const float integratedGauss[], int nGauss) {
        vec3f posLower = p - vec3f(radius);
        vec3f posUpper = p + vec3f(radius);

        vec3f voxelLower = getVoxelLower(voxel, bounds, voxelCount);
        vec3f voxelUpper = getVoxelUpper(voxel, bounds, voxelCount);

        vec3f linearOverlap = vec3f((min(voxelUpper.x, posUpper.x) - max(voxelLower.x, posLower.x)) / (2.0f * radius),
            (min(voxelUpper.y, posUpper.y) - max(voxelLower.y, posLower.y)) / (2.0f * radius),
            (min(voxelUpper.z, posUpper.z) - max(voxelLower.z, posLower.z)) / (2.0f * radius));

        // Still not clear yet if this is legal?
        return integratedGauss[min(int(linearOverlap.x * nGauss), nGauss - 1)]
            * integratedGauss[min(int(linearOverlap.y * nGauss), nGauss - 1)]
            * integratedGauss[min(int(linearOverlap.z * nGauss), nGauss - 1)];
    }

#pragma endregion

#pragma region Transformation
    vec3f DensityVolume::toDensityVolumeSpace(const vec3f vec, const vec3f x, const vec3f y, const vec3f z, const vec3f c, const bool direction) {
        if (direction)
            return vec.x * x + vec.y * y + vec.z * z;

        vec3f v = vec - c;
        return v.x * x + v.y * y + v.z * z;
    }
#pragma endregion

    void DensityVolume::buildDensityField(Model::SP model, const int n, const vec3f xUnit, const vec3f yUnit, const vec3f zUnit) {

        vec3f center = model->getBounds().center();

        vec3f r = vec3f(model->radius);

        box3f bounds = box3f(toDensityVolumeSpace(model->particles[0].pos - r, xUnit, yUnit, zUnit, center, false),
            toDensityVolumeSpace(model->particles[0].pos + r, xUnit, yUnit, zUnit, center, false));

        // Build axis aligned bounding box in view aligned space
        for (int i = 1; i < model->particles.size(); i++) {
            bounds.extend(toDensityVolumeSpace(model->particles[i].pos - r, xUnit, yUnit, zUnit, center, false));
            bounds.extend(toDensityVolumeSpace(model->particles[i].pos + r, xUnit, yUnit, zUnit, center, false));
        }

        vec3f boundsSize = bounds.upper - bounds.lower;


        // Make voxels as cube-like as possible
        vec3i voxelCount;
        if (boundsSize.x < boundsSize.y && boundsSize.x < boundsSize.z)
            voxelCount = vec3i(n, float(n) * (boundsSize.y / boundsSize.x), float(n) * (boundsSize.z / boundsSize.x));
        else if (boundsSize.y < boundsSize.x && boundsSize.y < boundsSize.z)
            voxelCount = vec3i(float(n) * (boundsSize.x / boundsSize.y), n, float(n) * (boundsSize.z / boundsSize.y));
        else
            voxelCount = vec3i(float(n) * (boundsSize.x / boundsSize.z), float(n) * (boundsSize.y / boundsSize.z), n);

        std::cout << "building density field: " << voxelCount.x << " x " << voxelCount.y << " x " << voxelCount.z << std::endl;

        vec3f cellSize = vec3f(boundsSize.x / voxelCount.x, boundsSize.y / voxelCount.y, boundsSize.z / voxelCount.z);
        float cellVolume = cellSize.x * cellSize.y * cellSize.z;

        // Set density context, i.e. bounds of the covered volume and voxel counts along each dimension
        densityContext = std::vector<vec3f>(7);
        densityContext[0] = bounds.lower;
        densityContext[1] = bounds.upper;
        densityContext[2] = vec3f(voxelCount.x, voxelCount.y, voxelCount.z);

        densityContext[3] = xUnit;    // x unit vector in density volume space
        densityContext[4] = yUnit;    // y unit vector in density volume space
        densityContext[5] = zUnit;    // z unit vector in density volume space
        densityContext[6] = center;

        particleDensity = std::vector<float>(voxelCount.x * voxelCount.y * voxelCount.z, 0);

        // Integrated gauss lookup table
        const int nGauss = 256;
        const float zAlpha = 1.5f;
        const float mu = 0.0f;
        const float sigma = 0.44f;
        float integratedGauss[nGauss];
        float sum = 0.0f;
        for (int i = 0; i < nGauss; i++) {
            sum += gauss((float(i) / float(nGauss)) * 2.0f * zAlpha - zAlpha, mu, sigma);
            integratedGauss[i] = sum; 
        }
        for (int i = 0; i < nGauss; i++) { //Normalize
            integratedGauss[i] /= sum;
        }

        // Splat particles into voxels
        for (int i = 0; i < model->particles.size(); i++) {
            Particle& p = model->particles[i];
            vec3f pos = toDensityVolumeSpace(p.pos, xUnit, yUnit, zUnit, center, false);

            box3i boundingVoxels = getBoundingVoxels(pos, model->radius, bounds, voxelCount);
            for (int x = boundingVoxels.lower.x; x <= boundingVoxels.upper.x; x++) {
                for (int y = boundingVoxels.lower.y; y <= boundingVoxels.upper.y; y++) {
                    for (int z = boundingVoxels.lower.z; z <= boundingVoxels.upper.z; z++) {
                        vec3i voxel = vec3i(x, y, z);
                        float overlap = getOverlap(pos, model->radius, voxel, bounds, voxelCount, integratedGauss, nGauss);
                        particleDensity[voxelCount.y * voxelCount.z * voxel.x + voxelCount.z * voxel.y + voxel.z] += overlap;
                    }
                }
            }           
        }

        // Normalize Density
        for (int i = 0; i < particleDensity.size(); i++) {
            particleDensity[i] /= cellVolume;
        }
        return;
    }
}