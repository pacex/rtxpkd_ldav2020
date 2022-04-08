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

#include "rtxParticles/treelets/Treelets.h"
#include "rtxParticles/common/PKD.h"
#include "rtxParticles/treelets/programs/treelets.h"
// std
#include <mutex>
#include <functional>


namespace pkd {

  extern "C" const char embedded_treelets_programs[];

  using pkd::device::TreeletsGeomData;

  TreeletParticles::TreeletParticles()
  {
    module = owlModuleCreate(context,embedded_treelets_programs);
  }


  size_t sort_partition(Model::SP model, size_t begin, size_t end, box3f bounds, int& splitDim)
  {
    // -------------------------------------------------------
    // determine split pos
    // -------------------------------------------------------
    splitDim = arg_max(bounds.span());
    float splitPos = bounds.center()[splitDim];

    // -------------------------------------------------------
    // now partition ...
    // -------------------------------------------------------
    size_t mid = begin;
    size_t l = begin, r = (end - 1);
    // quicksort partition:
    while (l <= r) {
      while (l < r && model->particles[l].pos[splitDim] < splitPos) ++l;
      while (l < r && model->particles[r].pos[splitDim] >= splitPos) --r;
      if (l == r) { mid = l; break; }

      std::swap(model->particles[l], model->particles[r]);
    }

    // catch-all for extreme cases where all particles are on the same
    // spot, and can't be split:
    if (mid == begin || mid == end) mid = (begin + end) / 2;

    return mid;
  }


  

  /*! todo: make this a cmd-line parameter, so we can run scripts to
    measure perf impact per size (already made it a static, so we
    can set it from main() before class is created */
  //int TreeletParticles::maxTreeletSize = 1000;

  template<typename MakeLeafLambda>
  void partitionRecursively(Model::SP model, size_t begin, size_t end,
                            const MakeLeafLambda &makeLeaf)
  {
    if (makeLeaf(begin,end,false))
      // could make into a leaf, done.
      return;

    // -------------------------------------------------------
    // parallel bounding box computation
    // -------------------------------------------------------
    box3f bounds;
    std::mutex boundsMutex;
    parallel_for_blocked(begin,end,32*1024,[&](size_t blockBegin, size_t blockEnd){
        box3f blockBounds;
        for (size_t i=blockBegin;i<blockEnd;i++) 
          blockBounds.extend(model->particles[i].pos);
        std::lock_guard<std::mutex> lock(boundsMutex);
        bounds.extend(blockBounds);
      });

    int splitDim;
    auto mid = sort_partition(model, begin, end, bounds, splitDim);

    // -------------------------------------------------------
    // and recurse ...
    // -------------------------------------------------------
    parallel_for(2,[&](int side){
        if (side)
          partitionRecursively(model,begin,mid,makeLeaf);
        else
          partitionRecursively(model,mid,end,makeLeaf);
      });
  }
  
  std::vector<PKDlet> prePartition_inPlace(Model::SP model, size_t maxSize)
  {
    std::mutex resultMutex;
    std::vector<PKDlet> result;

    partitionRecursively(model,0ULL,model->particles.size(),[&](size_t begin, size_t end, bool force) {
        /*bool makeLeaf() :*/
        const size_t size = end - begin;
        if (size > maxSize && !force) return false;

        PKDlet treelet;
        treelet.begin  = begin;
        treelet.end    = end;
        treelet.bounds = box3f();
        for (size_t i=begin;i<end;i++) {
          treelet.bounds.extend(model->particles[i].pos-model->radius);
          treelet.bounds.extend(model->particles[i].pos+model->radius);
        }
        
        std::lock_guard<std::mutex> lock(resultMutex);
        result.push_back(treelet);
        return true;
      });
    
    return std::move(result);
  }


  box3f getBounds(std::vector<pkd::Particle> const& particles, size_t begin, size_t end)
  {
    box3f bounds;
    for (size_t i = begin; i < end; ++i) {
      bounds.extend(particles[i].pos);
    }
    return bounds;
  }

  
  void TreeletParticles::buildModel(Model::SP model, bool override_model)
  {
    /*! a pair of int (begin,end) per treelet */
    std::cout << "#pkd.treelets: pre-partition into blocks of at most "
              << maxTreeletSize << " particles" << std::endl;

    std::vector<PKDlet> treelets
      = prePartition_inPlace(model,maxTreeletSize);
    

    std::cout << "#pkd.treelets: created " << treelets.size()
              << " blocks, now turn them into PKDs" << std::endl;
    parallel_for(treelets.size(),[&](size_t treeletID){
        makePKD(model->particles, treelets[treeletID].begin, treelets[treeletID].end, treelets[treeletID].bounds);
      });
    
    std::cout << "#pkd.treelets: on average " << prettyNumber(model->particles.size() / treelets.size()) << " per treelet" << std::endl;
    std::cout << "#pkd.treelets: treelets created, uploading to optix" << std::endl;
    // ------------------------------------------------------------------
    // upload begin/end buffer for treelets (for now, do ints)
    // ------------------------------------------------------------------
        
    treeletBuffer
      = owlDeviceBufferCreate(context,
      OWL_USER_TYPE(PKDlet),
      treelets.size(),
      treelets.data());
      
    

    // ------------------------------------------------------------------
    // now, all the other setup
    // ------------------------------------------------------------------
    
    // first, we need a buffer that contains all the
    // particles.... create and upload

    particleBuffer
      = owlDeviceBufferCreate(context,
      OWL_USER_TYPE(model->particles[0]),
      model->particles.size(),
      model->particles.data());
    
    // with those particles, we can now craete a user geometry with
    // appropriate intersection and bounds programs, and assign the
    // buffer

    OWLVarDecl treeletsVars[] = {
      { "treeletBuffer",  OWL_BUFPTR, OWL_OFFSETOF(TreeletsGeomData,treeletBuffer)},
      { "particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(TreeletsGeomData,particleBuffer)},
      { "particleRadius", OWL_FLOAT , OWL_OFFSETOF(TreeletsGeomData,particleRadius)},
      { /* sentinel to mark end of list */ }
    };

    PING;
    OWLGeomType treeletsType
      = owlGeomTypeCreate(context,
                          OWL_GEOMETRY_USER,
                          sizeof(TreeletsGeomData),
                          treeletsVars,-1);
    owlGeomTypeSetBoundsProg(treeletsType,module,
                             "treelet_bounds");
    if (bruteForce)
      owlGeomTypeSetIntersectProg(treeletsType,0,module,
                                  "treelet_brute_intersect");
    else
      owlGeomTypeSetIntersectProg(treeletsType,0,module,
                                  "treelet_intersect");
    owlGeomTypeSetClosestHit(treeletsType,0,module,
                             "treelet_closest_hit");
    
    // optix::Geometry geom = context->createGeometry();
    PING;
    OWLGeom geom = owlGeomCreate(context,treeletsType);
    owlGeomSetPrimCount(geom,treelets.size());
    
    PING;
    owlGeomSetBuffer(geom,"particleBuffer",particleBuffer);
    owlGeomSet1f(geom,"particleRadius",model->radius);
    owlGeomSetBuffer(geom,"treeletBuffer",treeletBuffer);
    
    // now that we have a geom, create a material to render it with.
    
    // now have a GI, put it into a ggroup, so we can assign an accel
    // and trace rays against
    // optix::GeometryGroup gg = context->createGeometryGroup();
    owlBuildPrograms(context);
    OWLGroup ug = owlUserGeomGroupCreate(context, 1, &geom);
    owlGroupBuildAccel(ug);
    PING;
    this->world = owlInstanceGroupCreate(context, 1, &ug);
    owlGroupBuildAccel(this->world);
    PING;
  }

  void TreeletParticles::resizeFrameBuffer(const vec2i& newSize) {
      OptixParticles::resizeFrameBufferGeneral(newSize);
  }

}
