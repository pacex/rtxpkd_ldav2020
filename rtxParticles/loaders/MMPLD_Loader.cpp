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

// ours
#include "rtxParticles/loaders/MMPLD_Loader.h"
// external: MMPLD
#include "mmpld.h"

namespace pkd {

#ifdef HAVE_MMPLD
 vec3f getMMPLDParticle(const char* baseptr, mmpld::vertex_type const& vt)
 {
    vec3f ret;

    if (vt == mmpld::vertex_type::FLOAT_XYZ || vt == mmpld::vertex_type::FLOAT_XYZR) {
      auto const fptr = reinterpret_cast<float const*>(baseptr);
      ret = vec3f(fptr[0], fptr[1], fptr[2]);
    } else if (vt == mmpld::vertex_type::DOUBLE_XYZ) {
      auto const dptr = reinterpret_cast<double const*>(baseptr);
      ret = vec3f(dptr[0], dptr[1], dptr[2]);
    }

    return ret;
  }

  void normalizeMMPLDParticle(vec3f& pos, std::array<float, 6> const& bbox)
  {
    pos.x -= bbox[0];
    pos.x /= bbox[3]-bbox[0];
    pos.y -= bbox[1];
    pos.y /= bbox[4]-bbox[1];
    pos.z -= bbox[2];
    pos.z /= bbox[5]-bbox[2];
  }
#endif
    
  
  Model::SP loadMMPLD(const std::string &fileName)
  {
    Model::SP model = std::make_shared<Model>();

    mmpld::frame_t frame;
    std::array<float, 6> bbox;

    {
        mmpld::mmpld file(fileName);

        frame = file.ReadFrame(0);

        bbox = file.GetBBox();
    }

    auto pl_count = frame.data.size();
       
    for (size_t plidx = 0; plidx < pl_count; ++plidx) {
      auto const& entry = frame.data[plidx];

      // TODO radius per list entry
      model->radius = std::max(model->radius, entry.list_header.global_radius);

      auto const& vt = entry.list_header.vert_type;
      if (!mmpld::HasData(vt)) continue;

      auto const pcount = entry.list_header.particle_count;
      auto const stride = entry.vertex_stride + entry.color_stride;

      for (size_t pidx = 0; pidx < pcount; ++pidx) {
        Particle part;

        auto base_ptr = entry.data.data()+pidx*stride;

        part.pos = getMMPLDParticle(base_ptr, vt);
        model->particles.push_back(part);
      }
    }
    return model;
  }
  
}
