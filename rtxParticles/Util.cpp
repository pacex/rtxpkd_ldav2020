
#include "Util.h"

namespace pkd {


	void Util::invertColumnMat3(vec3f& a, vec3f& b, vec3f& c) {
		vec3f newA, newB, newC;

		float det = a.x * (b.y * c.z - b.z * c.y) -
			b.x * (a.y * c.z - c.y * a.z) +
			c.x * (a.y * b.z - b.y * a.z);

		float invdet = 1 / det;

		newA.x = (b.y * c.z - b.z * c.y) * invdet;
		newB.x = (c.x * b.z - b.x * c.z) * invdet;
		newC.x = (b.x * c.y - c.x * b.y) * invdet;
		newA.y = (c.y * a.z - a.y * c.z) * invdet;
		newB.y = (a.x * c.z - c.x * a.z) * invdet;
		newC.y = (a.y * c.x - a.x * c.y) * invdet;
		newA.z = (a.y * b.z - a.z * b.y) * invdet;
		newB.z = (a.z * b.x - a.x * b.z) * invdet;
		newC.z = (a.x * b.y - a.y * b.x) * invdet;

		a = newA;
		b = newB;
		c = newC;

		return;
	}

}