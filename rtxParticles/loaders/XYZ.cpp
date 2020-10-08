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

// gdt
#include "rtxParticles/loaders/XYZ.h"

namespace pkd {

  /*! fiven a file name of "<intN>_<floatRadius>.random", this creates
      a random model with N particles of given radius */
  Model::SP loadXYZ(const std::string &fileName)
  {
    Model::SP model = std::make_shared<Model>();
    model->radius = 100.f;
    FILE *file = fopen(fileName.c_str(),"rb");
    if (!file) throw std::runtime_error("could not open file '"+fileName+"'");
    while (!feof(file)) {
      vec3f p,v;
      Particle part;
      //part.matID = -1;
      int numRead = fscanf(file,"%f %f %f %f %f %f\n",
                           &p.x,&p.y,&p.z,
                           &v.x,&v.y,&v.z);
      if (numRead == 3) {
        part.pos = p;
        model->particles.push_back(part);
      } else if (numRead == 6) {
        part.pos = p;
        model->particles.push_back(part);
      }
    }
    fclose(file);
    return model;
  }
  
}
