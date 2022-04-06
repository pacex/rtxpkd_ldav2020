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
	static class Util {

    public:
        static void invertColumnMat3(vec3f& a, vec3f& b, vec3f& c);
	};
}