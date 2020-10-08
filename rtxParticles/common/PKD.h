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

#pragma once

// ours
#include "rtxParticles/common/Particles.h"
#include "owl/common/parallel/parallel_for.h"

namespace pkd {
  
  inline size_t lChild(size_t P) { return 2*P+1; }
  inline size_t rChild(size_t P) { return 2*P+2; }
  
  template<class Comp>
  inline void trickle(const Comp &worse, size_t P, Particle *particle, size_t N, int dim)
  {
    if (P >= N) return;
    
    while (1) {
      const size_t L = lChild(P);
      const size_t R = rChild(P);
      const bool lValid = (L < N);
      const bool rValid = (R < N);

      if (!lValid) return;
      size_t C = L;
      if (rValid && worse(particle[R].pos[dim],
                          particle[L].pos[dim]))
        C = R;
      
      if (!worse(particle[C].pos[dim],
                 particle[P].pos[dim]))
        return;

      std::swap(particle[C],particle[P]);
      P = C;
    }
  }

  template<class Comp>
  inline void makeHeap(const Comp &comp, size_t P, Particle *particle, size_t N, int dim)
  {
    if (P >= N) return;
    const size_t L = lChild(P);
    const size_t R = rChild(P);
    makeHeap(comp,L,particle,N,dim);
    makeHeap(comp,R,particle,N,dim);
    trickle(comp,P,particle,N,dim);
  }

  inline void recBuild(size_t /* root node */P,
      Particle *particle,
      size_t N,
      box3f bounds)
  {
      if (P >= N) return;

      int dim = arg_max(bounds.span());

      const size_t L = lChild(P);
      const size_t R = rChild(P);
      const bool lValid = (L < N);
      const bool rValid = (R < N);
      makeHeap(std::greater<float>(), L, particle, N, dim);
      makeHeap(std::less<float>(), R, particle, N, dim);

      if (rValid) {
          while (particle[L].pos[dim] > particle[R].pos[dim]) {
              std::swap(particle[L], particle[R]);
              trickle(std::greater<float>(), L, particle, N, dim);
              trickle(std::less<float>(), R, particle, N, dim);
          }
          if (particle[L].pos[dim] > particle[P].pos[dim]) {
              std::swap(particle[L], particle[P]);
              particle[L].dim = dim;
          }
          else if (particle[R].pos[dim] < particle[P].pos[dim]) {
              std::swap(particle[R], particle[P]);
              particle[R].dim = dim;
          }
          else
              /* nothing, root fits */;
      }
      else if (lValid) {
          if (particle[L].pos[dim] > particle[P].pos[dim]) {
              std::swap(particle[L], particle[P]);
              particle[L].dim = dim;
          }
      }

      box3f lBounds = bounds;
      box3f rBounds = bounds;
      lBounds.upper[dim] = rBounds.lower[dim] = particle[P].pos[dim];
      particle[P].dim = dim;

      parallel_for(2, [&](int childID) {
          if (childID) {
              recBuild(L, particle, N, lBounds);
          }
          else {
              recBuild(R, particle, N, rBounds);
          }
          });
      
  }


  inline void checkTree(const size_t P, const Particle *const particle, const size_t N, const int dim,
                 const box3f &bounds)
  {
    if (P >= N) return;

    
    const Particle &pp = particle[P];
    if (!bounds.contains(pp.pos)) {
        std::cout << "Tree validation failed at " << P << "\n" << pp.pos << "\n" << bounds << std::endl;
        throw std::runtime_error("pkd tree validation failed!!!");
    }
    
    box3f lBounds = bounds;
    box3f rBounds = bounds;
    lBounds.upper[pp.dim] = rBounds.lower[pp.dim] = pp.pos[pp.dim];
    checkTree(lChild(P),particle,N,(dim+1)%3,lBounds);
    checkTree(rChild(P),particle,N,(dim+1)%3,rBounds);
  }

  // /*! build pkd for an _entire_ std::vector */
  
  inline void makePKD(std::vector<Particle> &particles, box3f bounds)
  {
      recBuild(/*node:*/0, particles.data(), particles.size(),bounds);
#ifdef CHECK
      checkTree(0, particles.data(), particles.size(), 0, bounds);
#endif
  }

  /*! make a pkd treelet _inside_ a particle array, for the given
      begin/end range. ie, the particle at position 'begin' will be
      the root node of a tree of (end-ebgin) paricles. Note this is
      _NOT_ the same as a pkd subtree rooted at position begin (for
      the latter case, the children of that node would be at 2*begin+1
      and 2*begin+2, while for the complete tree built by this
      function they'll be at begin+1 and begin+2. */
  
  inline void makePKD(std::vector<Particle> &particles, size_t begin, size_t end, box3f bounds)
  {
      recBuild(/*node:*/0, particles.data() + begin, end - begin, bounds);
#ifdef CHECK
      checkTree(0, particles.data() + begin, end - begin, 0, bounds);
#endif
  }

  inline void checkPKD(std::vector<Particle> &particles)
  {
#ifdef NDEBUG
    std::cout << "no tree validation in NDEBUG mode" << std::endl;
#else
    box3f bounds;
    for (auto p : particles)
      bounds.extend(p.pos);
    checkTree(/*node:*/0,particles.data(),particles.size(),/*dim*/0,
              bounds);
    std::cout << "Tree validation passed!" << std::endl;
#endif
  }

}
