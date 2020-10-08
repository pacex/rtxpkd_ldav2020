// ======================================================================== //
// Copyright 2019-2020 VISUS - University of Stuttgart                      //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//   http://www.apache.org/licenses/LICENSE-2.0                             //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "rtxParticles/loaders/PKD_Loader.h"

#include <fstream>

namespace pkd {
    struct IPart {
        vec3f pos;
        int dim;
    };

    Model::SP loadPKD(const std::string& fileName)
    {
        Model::SP model = std::make_shared<Model>();

        std::ifstream file(fileName, std::ios::binary);

        if (file.is_open()) {
            file.ignore(std::numeric_limits<std::streamsize>::max());
            auto length = file.gcount();
            file.clear();
            file.seekg(0, std::ios::beg);

            file.read(reinterpret_cast<char*>(&model->radius), sizeof(float));
            auto buffer_length = length - sizeof(float);
            std::vector<IPart> temp(buffer_length / sizeof(Particle));
            model->particles.resize(buffer_length / sizeof(Particle));
            //file.read(reinterpret_cast<char*>(model->particles.data()), buffer_length);
            file.read(reinterpret_cast<char*>(temp.data()), buffer_length);
            for (size_t i = 0; i < buffer_length / sizeof(Particle); ++i) {
                model->particles[i].pos = temp[i].pos;
                model->particles[i].dim = temp[i].dim;
            }
        }

        return model;
    }
}
