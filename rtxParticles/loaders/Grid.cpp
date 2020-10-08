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
#include "rtxParticles/loaders/Random.h"
// std
#include <random>

namespace pkd {
  
  static inline vec3f warp(const vec3f v)
  {
    return vec3f(cosf(6*v.x),
                 cosf(12*v.y+11*v.x),
                 sinf(8.6*v.z));
  }

  /*! fiven a file name of "<intN>_<floatRadius>.random", this creates
      a random model with N particles of given radius */
  Model::SP loadGrid(const std::string &fileName)
  {
    auto rand_dist = std::uniform_real_distribution<float>();
    auto rand_gen = std::mt19937();
    
    int nx,ny,nz;
    int numParsed = sscanf(fileName.c_str(),"%i_%i_%i.grid",
                           &nx,&ny,&nz);
    if (numParsed != 3)
      throw std::runtime_error("could not parse "+fileName+" as %i_%i_i.grid");
    
    Model::SP model = std::make_shared<Model>();
    for (int iz=0;iz<nz;iz++)
      for (int iy=0;iy<ny;iy++)
        for (int ix=0;ix<nx;ix++) {
          Particle part;
          part.pos
            = vec3f(ix,iy,iz)
            // + .4f*vec3f(rand_dist(rand_gen),rand_dist(rand_gen),rand_dist(rand_gen))
            ;
          //part.matID = -1;// ix+nx*iy+nx*ny*iz;//-1;
          model->particles.push_back(part);
        }
    return model;
  }
  
}
