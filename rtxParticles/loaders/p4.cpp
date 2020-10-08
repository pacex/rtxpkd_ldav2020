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

#include "rtxParticles/loaders/XYZ.h"
#include <fstream>

namespace pkd {

  /*! fiven a file name of "<intN>_<floatRadius>.random", this creates
      a random model with N particles of given radius */
  Model::SP loadP4(const std::string &fileName)
  {
    Model::SP model = std::make_shared<Model>();
    model->radius = 100.f;
    std::ifstream file(fileName,std::ios::binary);
    if (!file.good()) throw std::runtime_error("could not open file '"+fileName+"'");
    struct {
      vec3f pos;
      int   other;
    } p4;
    while (file.read((char*)&p4,sizeof(p4))) {
      Particle part;
      part.pos   = p4.pos;
      //part.matID = p4.other;
      model->particles.push_back(part);
    }
    return model;
  }
}
