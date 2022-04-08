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

#include "rtxParticles/noPKD/NoPKD.h"
#include "rtxParticles/noPKD/programs/noPKD.h"

namespace pkd {
  
  extern "C" const char embedded_noPKD_programs[];
  
  using device::NoPKDGeomData;
  
  NoPKDParticles::NoPKDParticles()
  {
    module = owlModuleCreate(context,embedded_noPKD_programs);
  }

  /*! the setup code for the noPKD method - todo: move this in a
    derived class */
  void NoPKDParticles::buildModel(Model::SP model, bool override_model)
  {
    // // first, we need a buffer that contains all the
    // // particles.... create and upload
    
    particleBuffer
      = owlDeviceBufferCreate(context,
      OWL_USER_TYPE(model->particles[0]),
      model->particles.size(),
      model->particles.data());


    OWLVarDecl noPKDVars[] = {
      { "particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(NoPKDGeomData,particleBuffer)},
      { "particleRadius", OWL_FLOAT , OWL_OFFSETOF(NoPKDGeomData,particleRadius)},
      { /* sentinel to mark end of list */ }
    };

    PING;
    OWLGeomType noPKDType
      = owlGeomTypeCreate(context,
                          OWL_GEOMETRY_USER,
                          sizeof(NoPKDGeomData),
                          noPKDVars,-1);
    owlGeomTypeSetBoundsProg(noPKDType,module,
                             "noPKD_bounds");
    owlGeomTypeSetIntersectProg(noPKDType,0,module,
                                "noPKD_intersect");
    owlGeomTypeSetClosestHit(noPKDType,0,module,
                             "noPKD_closest_hit");

    OWLGeom geom = owlGeomCreate(context,noPKDType);
    owlGeomSetPrimCount(geom,model->particles.size());
    
    // with those particles, we can now craete a user geometry with
    // appropriate intersection and bounds programs, and assign the
    // buffer
    
    owlGeomSetBuffer(geom,"particleBuffer",particleBuffer);
    owlGeomSet1f(geom,"particleRadius",model->radius);
    
    owlBuildPrograms(context);

    OWLGroup ug = owlUserGeomGroupCreate(context, 1, &geom);
    owlGroupBuildAccel(ug);
    this->world = owlInstanceGroupCreate(context, 1, &ug);
    PING;
    owlGroupBuildAccel(this->world);
    PING;
  }

  void NoPKDParticles::resizeFrameBuffer(const vec2i& newSize) {
      OptixParticles::resizeFrameBufferGeneral(newSize);
  }
  
}
