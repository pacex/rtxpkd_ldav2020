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

#include "rtxParticles/allPKD/AllPKD.h"
#include "rtxParticles/allPKD/programs/allPKD.h"
#include "rtxParticles/common/PKD.h"

namespace pkd {

  extern "C" const char embedded_allPKD_programs[];
  
  using device::AllPKDGeomData;
  
  AllPKDParticles::AllPKDParticles()
  {
    module = owlModuleCreate(context,embedded_allPKD_programs);
  }

  /*! the setup code for the noPKD method - todo: move this in a
    derived class */
  void AllPKDParticles::buildModel(Model::SP model, bool override_model)
  {
    std::cout << "#pkd: making model into a pkd... " << std::endl;
    const box3f bounds = model->getBounds();
    if (!override_model) {
        makePKD(model->particles, bounds);
        std::cout << "#pkd: done building pkd tree... " << std::endl;
    }
    
    particleBuffer
      = owlDeviceBufferCreate(context,
      OWL_USER_TYPE(model->particles[0]),
      model->particles.size(),
      model->particles.data());

    OWLVarDecl allPKDVars[] = {
      { "particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(AllPKDGeomData,particleBuffer)},
      { "particleRadius", OWL_FLOAT , OWL_OFFSETOF(AllPKDGeomData,particleRadius)},
      { "particleCount",  OWL_INT,    OWL_OFFSETOF(AllPKDGeomData,particleCount)},
      { "bounds.lower",   OWL_FLOAT3, OWL_OFFSETOF(AllPKDGeomData,worldBounds.lower)},
      { "bounds.upper",   OWL_FLOAT3, OWL_OFFSETOF(AllPKDGeomData,worldBounds.upper)},

      { "depthConfidenceAccumBuffer",   OWL_BUFPTR, OWL_OFFSETOF(AllPKDGeomData,depthConfidenceAccumBufferPtr)},
      { "depthConfidenceCullBuffer",   OWL_BUFPTR, OWL_OFFSETOF(AllPKDGeomData,depthConfidenceCullBufferPtr)},
      { "confidentDepthBuffer",   OWL_BUFPTR, OWL_OFFSETOF(AllPKDGeomData,confidentDepthBufferPtr)},
      { "frameStateBuffer",   OWL_BUFPTR, OWL_OFFSETOF(AllPKDGeomData,frameStateBuffer)},
      { "accumIDLastCulled",   OWL_BUFPTR, OWL_OFFSETOF(AllPKDGeomData,accumIDLastCulled)},
      { "densityBuffer",   OWL_BUFPTR, OWL_OFFSETOF(AllPKDGeomData,densityBuffer)},
      { "densityContextBuffer",   OWL_BUFPTR, OWL_OFFSETOF(AllPKDGeomData,densityContextBuffer)},
      { "normalCdfBuffer",   OWL_BUFPTR, OWL_OFFSETOF(AllPKDGeomData,normalCdfBuffer)},
      { /* sentinel to mark end of list */ }
    };

    OWLGeomType allPKDType
      = owlGeomTypeCreate(context,
                          OWL_GEOMETRY_USER,
                          sizeof(AllPKDGeomData),
                          allPKDVars,-1);
    owlGeomTypeSetBoundsProg(allPKDType,module,
                             "allPKD_bounds");
    owlGeomTypeSetIntersectProg(allPKDType,0,module,
                                "allPKD_intersect");
    owlGeomTypeSetClosestHit(allPKDType,0,module,
                             "allPKD_closest_hit");

    this->geom = owlGeomCreate(context,allPKDType);
    // from the optix point of view, there is only *one* primitmive in
    // the geometry, the entire pkd tree.
    owlGeomSetPrimCount(geom,1); 
    
    owlGeomSetBuffer(geom,"particleBuffer",particleBuffer);
    owlGeomSet1f(geom,"particleRadius",model->radius);
    owlGeomSet1i(geom,"particleCount",(int)model->particles.size());
    owlGeomSet3f(geom,"bounds.lower",(const owl3f&)bounds.lower);
    owlGeomSet3f(geom,"bounds.upper",(const owl3f&)bounds.upper);

    
    owlGeomSetBuffer(geom, "depthConfidenceAccumBuffer", depthConfidenceAccumBuffer);
    owlGeomSetBuffer(geom, "depthConfidenceCullBuffer", depthConfidenceCullBuffer);
    owlGeomSetBuffer(geom, "confidentDepthBuffer", confidentDepthBuffer);
    owlGeomSetBuffer(geom, "frameStateBuffer", frameStateBuffer);
    owlGeomSetBuffer(geom, "accumIDLastCulled", accumIDLastCulled);
    owlGeomSetBuffer(geom, "densityBuffer", densityBuffer);
    owlGeomSetBuffer(geom, "densityContextBuffer", densityContextBuffer);
    owlGeomSetBuffer(geom, "normalCdfBuffer", normalCdfBuffer);
    
    owlBuildPrograms(context);

    OWLGroup ug = owlUserGeomGroupCreate(context, 1, &geom);
    owlGroupBuildAccel(ug);

    this->world = owlInstanceGroupCreate(context, 1, &ug);

    owlGroupBuildAccel(this->world);
  }

  void AllPKDParticles::resizeFrameBuffer(const vec2i& newSize) {

      OptixParticles::resizeFrameBufferGeneral(newSize);
      
      owlGeomSetBuffer(geom, "depthConfidenceAccumBuffer", depthConfidenceAccumBuffer);
      owlGeomSetBuffer(geom, "depthConfidenceCullBuffer", depthConfidenceCullBuffer);
      owlGeomSetBuffer(geom, "confidentDepthBuffer", confidentDepthBuffer);
      owlGeomSetBuffer(geom, "frameStateBuffer", frameStateBuffer);
      owlGeomSetBuffer(geom, "accumIDLastCulled", accumIDLastCulled);
      owlGeomSetBuffer(geom, "densityBuffer", densityBuffer);
      owlGeomSetBuffer(geom, "densityContextBuffer", densityContextBuffer);
      owlGeomSetBuffer(geom, "normalCdfBuffer", normalCdfBuffer);
      
  }
  
}
