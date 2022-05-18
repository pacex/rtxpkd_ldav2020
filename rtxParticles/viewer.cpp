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

// eventually to become a standalone library that does all rendering,
// but no i/o, viewer, etc:
#include "rtxParticles/common/Particles.h"
#include "rtxParticles/OptixParticles.h"
#include "rtxParticles/noPKD/NoPKD.h"
#include "rtxParticles/allPKD/AllPKD.h"
#include "rtxParticles/treelets/Treelets.h"

// eventually to go into 'apps/'
#include "rtxParticles/loaders/loaders.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "submodules/3rdParty/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION 1
#include "submodules/3rdParty/stb_image.h"

#include "Util.h"

#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
// gdt
// #include "gdt/math/vec.h"
// #include "gdt/math/box.h"
#include "owl/common/viewerWidget/ViewerWidget.h"
#include <GL/glut.h>
// std
#include <queue>
#include <map>
#include <set>
#include <string.h>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <fstream>

#define MEASURE_WARMUP_COUNT 1
#define MEASURE_FRAME_COUNT 20


namespace pkd {

    std::string method = "allPKD";

    int treelet_size = 1000;

    bool measure = false;
    std::string screenShotFileName = "screenshot.png";

    struct {
        struct {
            vec3f vp = vec3f(0.f);
            vec3f vu = vec3f(0.f);
            vec3f vi = vec3f(0.f);
        } camera;
        vec2i windowSize = vec2i(800, 800);

    } cmdline;


    float radius = 0.5f;
    void usage(const std::string& msg)
    {
        if (msg != "") std::cerr << "Error: " << msg << std::endl << std::endl;
        std::cout << "Usage: ./rtxParticleViewer <inputfile>" << std::endl;
        std::cout << "--camera pos.x pos.y pos.z at.x at.y at.z up.x up.y up.z" << std::endl;
        std::cout << "--size windowSize.x windowSize.y" << std::endl;
        std::cout << "--rec-depth <pathTracerRecursionDepth>" << std::endl;
        exit(msg != "");
    }

    struct ModelViewer : public owl::viewer::ViewerWidget
    {
        typedef ViewerWidget inherited;

        OptixParticles& particles;
        int renderBuffer; //Buffer to render: 0=col, 1=norm, 2=depth, 3=coverage
        bool logFps = false;

        ModelViewer(owl::viewer::GlutWindow::SP window,
            OptixParticles& particles)
            : ViewerWidget(window),
            particles(particles)
        {
            frameState.samplesPerPixel = 1;
            renderBuffer = 0;
        }

        // /*! this function gets called whenever the viewer widget changes camera settings */
        virtual void cameraChanged() override
        {
            inherited::cameraChanged();
            const viewer::SimpleCamera& camera = inherited::getCamera();

            const vec3f screen_du = camera.screen.horizontal / float(getWindowSize().x);
            const vec3f screen_dv = camera.screen.vertical / float(getWindowSize().y);
            frameState.camera_screen_du = screen_du;
            frameState.camera_screen_dv = screen_dv;
            frameState.camera_screen_00 = camera.screen.lower_left;
            frameState.camera_screen_center = camera.screen.lower_left + 0.5f * camera.screen.horizontal + 0.5f * camera.screen.vertical;
            frameState.camera_lens_center = camera.lens.center;
            frameState.camera_lens_du = camera.lens.du;
            frameState.camera_lens_dv = camera.lens.dv;
            frameState.accumID = 0;
            particles.updateFrameState(frameState);
            glutPostRedisplay();
        }

        /*! window notifies us that we got resized */
        virtual void resize(const vec2i& newSize) override
        {
            this->fbSize = newSize;

            particles.resizeFrameBuffer(newSize);
            std::cout << "rebuilding sbt because raygen buffers changed ..." << std::endl;
            owlBuildSBT(particles.context);

            // ... tell parent to resize (also resizes the pbo in the window)
            inherited::resize(newSize);

            // ... and finally: update the camera's aspect
            setAspect(newSize.x / float(newSize.y));

            // update camera as well, since resize changed both aspect and
            // u/v pixel delta vectors ...
            updateCamera();
        }

        /*! gets called whenever the viewer needs us to re-render out widget */
        virtual void render() override
        {
            if (fbSize.x < 0) return;
            // static double t_last = -1;

            if (measure) {
                static int g_frameID = 0;
                int frameID = g_frameID++;
                if (frameID == MEASURE_WARMUP_COUNT) {

                    std::cout << "#measure: starting..." << std::endl;

                    auto const t_start = std::chrono::high_resolution_clock::now();

                    for (int i = 0; i < MEASURE_FRAME_COUNT; ++i) {
                        particles.render();
                        frameState.accumID++;
                        particles.updateFrameState(frameState);
                    }
                    auto const t_end = std::chrono::high_resolution_clock::now();
                    auto const duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

                    screenShot();
                    std::cout << "#measure: renderered " << MEASURE_FRAME_COUNT
                        << " frames in " << duration.count() << "ms" << std::endl;

                    printf("AVG_FPS %.1f \n", (MEASURE_FRAME_COUNT / (duration.count() / 1000.0f)));
                    std::cout << "#measure: finished." << std::endl;
                    //exit(0);
                    measure = false;
                    g_frameID = 0;
                }
            }


            particles.render();


            uint32_t* fb;

            switch (renderBuffer) {
            case 1:
                fb = particles.mapNormalBuffer();
                break;
            case 2:
                fb = particles.mapDepthBuffer();
                break;
            case 3:
                fb = particles.mapCoverageBuffer();
                break;
            default:
                fb = particles.mapColorBuffer();
                break;
            }

            if (!fb) return;

            window->drawPixels(fb);

            switch (renderBuffer) {
            case 1:
                particles.unmapNormalBuffer();
                break;
            case 2:
                particles.unmapDepthBuffer();
                break;
            case 3:
                particles.unmapCoverageBuffer();
                break;
            default:
                particles.unmapColorBuffer();
                break;
            }
            
            window->swapBuffers();


            static double t_last = 0.;
            static float fps = 0.f;
            double t_now = getCurrentTime();
            if (t_last != 0.) {
                double thisFPS = 1. / (t_now - t_last);
                if (fps == 0.f)
                    fps = thisFPS;
                else
                    fps = 0.9f * fps + 0.1f * thisFPS;
                char newTitle[1000];
                sprintf(newTitle, "rtxPKD (%3.1ffps)", fps);
                window->setTitle(newTitle);

                if (logFps) {
                    std::cout << thisFPS << std::endl;
                }
            }
            t_last = getCurrentTime();


            frameState.accumID++;
            particles.updateFrameState(frameState);
        }

        /*! this gets called when the user presses a key on the keyboard ... */
        virtual void key(char key, const vec2i& where)
        {
            switch (key) {
            case 'V':
                displayFPS = !displayFPS;
                break;
            case '<':
                frameState.heatMapScale *= 1.5f;
                frameState.accumID = 0;
                particles.updateFrameState(frameState);
                break;
            case '>':
                frameState.heatMapScale /= 1.5f;
                frameState.accumID = 0;
                particles.updateFrameState(frameState);
                break;

            case '^':
                frameState.dbgPixel = where;
                frameState.accumID = 0;
                particles.updateFrameState(frameState);
                break;

            case 'h':
            case 'H':
                frameState.heatMapEnabled ^= 1;
                PRINT((int)frameState.heatMapEnabled);
                frameState.accumID = 0;
                particles.updateFrameState(frameState);
                break;
            case 'C': {
                auto& fc = fullCamera;
                std::cout << "(C)urrent camera:" << std::endl;
                std::cout << "- from :" << fc.position << std::endl;
                std::cout << "- poi  :" << fc.getPOI() << std::endl;
                std::cout << "- upVec:" << fc.upVector << std::endl;
                std::cout << "- frame:" << fc.frame << std::endl;
                std::cout.precision(10);
                std::cout << "cmdline: --camera "
                    << fc.position.x << " "
                    << fc.position.y << " "
                    << fc.position.z << " "
                    << fc.getPOI().x << " "
                    << fc.getPOI().y << " "
                    << fc.getPOI().z << " "
                    << fc.upVector.x << " "
                    << fc.upVector.y << " "
                    << fc.upVector.z << std::endl;
            } break;
            case 'f': { // Toggle fps output
                logFps ^= 1;
            } break;
            case 'n': { // Switch displayed buffer
                renderBuffer++;
                renderBuffer = renderBuffer % 4;
            } break;
            case 'd': { // Transform density grid to align with current view
                auto& fc = fullCamera;
                vec3f baseX, baseY, baseZ;
                baseZ = normalize(fc.position - fc.getPOI());
                baseX = normalize(cross(baseZ, fc.upVector));
                baseY = normalize(cross(baseX, baseZ));

                Util::invertColumnMat3(baseX, baseY, baseZ);

                std::cout << "Rebuilding density field..." << std::endl;

                particles.buildDensityField(baseX,
                    baseY,
                    baseZ);

                std::cout << "Done." << std::endl;

                frameState.accumID = 0;
                particles.updateFrameState(frameState);
            } break;
            case 'q': { // Toggle depth quantisation
                frameState.quant ^= 1;
                frameState.accumID = 0;
                particles.updateFrameState(frameState);
            } break;
            case 'm': { // Measure performance
                measure = true;
            } break;
            case 'M': { // Reset accumulation and measure performance
                measure = true;
                frameState.probabilisticCulling ^= 1;
                frameState.accumID = 0;
                particles.updateFrameState(frameState);
            } break;
            case 'o': { // Toggle projection (culling only works correctly with ortho projection)
                frameState.orthoProjection ^= 1;
                frameState.accumID = 0;
                particles.updateFrameState(frameState);
            } break;
            case 'p': { // Toggle probabilistic culling
                frameState.probabilisticCulling ^= 1;
                frameState.accumID = 0;
                particles.updateFrameState(frameState);
                break;
            }
            case 'l': {
                std::cout << "Enter .ini file path:" << std::endl;
                std::string fname;
                std::cin >> fname;
                std::cout << "Enter view ID:" << std::endl;
                std::string section;
                std::cin >> section;


                frameState.samplesPerPixel = GetPrivateProfileInt("Properties", "spp", 1, fname.c_str());
                frameState.convergenceIterations = GetPrivateProfileInt("Properties", "conv-iter", 256, fname.c_str());
                frameState.kernelSize = GetPrivateProfileInt("Properties", "kernel-size", 0, fname.c_str());
                frameState.quant = (GetPrivateProfileInt("Properties", "quant", 0, fname.c_str()) == 1) ? true : false;
                frameState.interp = (GetPrivateProfileInt("Properties", "interp", 0, fname.c_str()) == 1) ? true : false;
                frameState.nBudget = GetPrivateProfileInt("Properties", "nBudget", 25, fname.c_str());

                std::string c_occ, xfrom, yfrom, zfrom, xto, yto, zto, xup, yup, zup;
                GetPrivateProfileString(section.c_str(), "xfrom", "0.0", const_cast<char*>(xfrom.c_str()), 16, fname.c_str());
                GetPrivateProfileString(section.c_str(), "yfrom", "0.0", const_cast<char*>(yfrom.c_str()), 16, fname.c_str());
                GetPrivateProfileString(section.c_str(), "zfrom", "0.0", const_cast<char*>(zfrom.c_str()), 16, fname.c_str());
                GetPrivateProfileString(section.c_str(), "xto", "0.0", const_cast<char*>(xto.c_str()), 16, fname.c_str());
                GetPrivateProfileString(section.c_str(), "yto", "0.0", const_cast<char*>(yto.c_str()), 16, fname.c_str());
                GetPrivateProfileString(section.c_str(), "zto", "0.0", const_cast<char*>(zto.c_str()), 16, fname.c_str());
                GetPrivateProfileString(section.c_str(), "xup", "0.0", const_cast<char*>(xup.c_str()), 16, fname.c_str());
                GetPrivateProfileString(section.c_str(), "yup", "1.0", const_cast<char*>(yup.c_str()), 16, fname.c_str());
                GetPrivateProfileString(section.c_str(), "zup", "0.0", const_cast<char*>(zup.c_str()), 16, fname.c_str());
                GetPrivateProfileString("Properties", "c_occ", "0.0", const_cast<char*>(c_occ.c_str()), 16, fname.c_str());
                
                frameState.c_occ = std::stof(c_occ);

                frameState.probabilisticCulling = false;

                setCameraOrientation(vec3f(std::stof(xfrom), std::stof(yfrom), std::stof(zfrom)),
                    vec3f(std::stof(xto), std::stof(yto), std::stof(zto)),
                    vec3f(std::stof(xup), std::stof(yup), std::stof(zup)),
                    70.0f);

                std::cout << "#testcase: loaded." << std::endl;

                break;
            }
            case '!': {
                screenShot();
            } break;
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                frameState.shadeMode = (key - '0');
                PRINT(frameState.shadeMode);
                frameState.accumID = 0;
                particles.updateFrameState(frameState);
                break;
            default:
                ViewerWidget::key(key, where);
            }
        }


        void screenShot()
        {
            const uint32_t* fb
                = (const uint32_t*)owlBufferGetPointer(particles.colorBuffer, 0);
            const vec2i fbSize = particles.fbSize;

            
            const std::string fileName = screenShotFileName;
            std::vector<uint32_t> pixels;
            for (int y = 0; y < fbSize.y; y++) {
                const uint32_t* line = fb + (fbSize.y - 1 - y) * fbSize.x;
                for (int x = 0; x < fbSize.x; x++) {
                    pixels.push_back(line[x] | (0xff << 24));
                }
            }

            stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
                pixels.data(), fbSize.x * sizeof(uint32_t));
            std::cout << "screenshot saved in '" << fileName << "'" << std::endl;
        }


        vec2i fbSize{ -1,-1 };
        device::FrameState frameState;
        bool displayFPS = true;
    };

    void dump_model(Model::SP model) {
        std::ofstream file("model.pkd", std::ios::binary);

        if (file.is_open()) {
            file.write(reinterpret_cast<char*>(&model->radius), sizeof(float));
            file.write(reinterpret_cast<char*>(model->particles.data()), sizeof(Particle) * model->particles.size());
        }
    }

    extern "C" int main(int argc, char** argv)
    {
        owl::viewer::GlutWindow::initGlut(argc, argv);

        std::string sceneFileName = "";

        int spp = 1;
        float c_occ = 0.95f;
        int voxelCount = 64;
        int convIter = 128;
        int nBudget = 25;
        int kernelSize = 0;
        bool debug = false;
        bool quant = false;
        bool interp = false;

        bool dumpModel = false;

        std::string ext = "";

        for (int i = 1; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg[0] != '-') {
                sceneFileName = arg;
                auto ext_start = sceneFileName.find_last_of('.');
                ext = sceneFileName.substr(ext_start);
            }
            else if (arg == "-o") {
                screenShotFileName = argv[++i];
            }
            else if (arg == "--dump") {
                dumpModel = true;
            }
            else if (arg == "--treelet16") {
                method = "treelets";
                treelet_size = 16;
            }
            else if (arg == "--brute16") {
                method = "brute";
                treelet_size = 16;
            }
            else if (arg == "--treelet4") {
                method = "treelets";
                treelet_size = 4;
            }
            else if (arg == "--treelet8") {
                method = "treelets";
                treelet_size = 8;
            }
            else if (arg == "--treelet16") {
                method = "treelets";
                treelet_size = 16;
            }
            else if (arg == "--treelet32") {
                method = "treelets";
                treelet_size = 32;
            }
            else if (arg == "--treelet64") {
                method = "treelets";
                treelet_size = 64;
            }
            else if (arg == "--brute64") {
                method = "brute";
                treelet_size = 64;
            }
            else if (arg == "--brute128") {
                method = "brute";
                treelet_size = 128;
            }
            else if (arg == "--treelet128") {
                method = "treelets";
                treelet_size = 128;
            }
            else if (arg == "--treelet256") {
                method = "treelets";
                treelet_size = 256;
            }
            else if (arg == "--treelet512") {
                method = "treelets";
                treelet_size = 512;
            }
            else if (arg == "--treelet1k") {
                method = "treelets";
                treelet_size = 1024;
            }
            else if (arg == "--treelet2k") {
                method = "treelets";
                treelet_size = 2048;
            }
            else if (arg == "--treelet3k") {
                method = "treelets";
                treelet_size = 3072;
            }
            else if (arg == "--treelet4k") {
                method = "treelets";
                treelet_size = 4096;
            }
            else if (arg == "--all-pkd" || arg == "allPKD" || arg == "--allpkd") {
                method = "allPKD";
            }
            else if (arg == "--no-pkd" || arg == "noPKD" || arg == "--nopkd") {
                method = "noPKD";
            }
            else if (arg == "--treelets" || arg == "treelets") {
                method = "treelets";
            }
            else if (arg == "--treelets-brute" || arg == "treelets-brute") {
                method = "treelets-brute";
            }
            else if (arg == "--method" || arg == "-m") {
                method = argv[++i];
            }
            else if (arg == "--spp" || arg == "-spp") {
                spp = std::atoi(argv[++i]);
            }
            else if (arg == "--max-size" || arg == "-ms") {
                treelet_size = std::atoi(argv[++i]);
            }
            else if (arg == "-win" || arg == "--size") {
                cmdline.windowSize.x = std::atoi(argv[++i]);
                cmdline.windowSize.y = std::atoi(argv[++i]);
            }
            else if (arg == "--treelet-size") {
                treelet_size = std::atoi(argv[++i]);
            }
            else if (arg == "--camera") {
                cmdline.camera.vp.x = std::atof(argv[++i]);
                cmdline.camera.vp.y = std::atof(argv[++i]);
                cmdline.camera.vp.z = std::atof(argv[++i]);
                cmdline.camera.vi.x = std::atof(argv[++i]);
                cmdline.camera.vi.y = std::atof(argv[++i]);
                cmdline.camera.vi.z = std::atof(argv[++i]);
                cmdline.camera.vu.x = std::atof(argv[++i]);
                cmdline.camera.vu.y = std::atof(argv[++i]);
                cmdline.camera.vu.z = std::atof(argv[++i]);
            }
            else if (arg == "--rec-depth") {
                OptixParticles::rec_depth = std::atoi(argv[++i]);
            }
            else if (arg == "--measure") {
                measure = true;
            }
            else if (arg == "--geometry") {
                cmdline.windowSize.x = std::atoi(argv[++i]);
                cmdline.windowSize.y = std::atoi(argv[++i]);
            }
            else if (arg == "--radius") {
                radius = std::atof(argv[++i]);
            }
            else if (arg == "--c_occ") {
                c_occ = std::atof(argv[++i]);
            }
            else if (arg == "--voxel-count") {
                OptixParticles::voxel_count = std::atof(argv[++i]);
            }
            else if (arg == "--n-budget") {
                nBudget = std::atof(argv[++i]);
            }
            else if (arg == "--conv-iter") {
                convIter = std::atof(argv[++i]);
            }
            else if (arg == "--debug") {
                debug = true;
            }
            else if (arg == "--quant") {
                quant = true;
            }
            else if (arg == "--interp") {
                interp = true;
            }
            else if (arg == "--kernel-size") {
                kernelSize = std::atof(argv[++i]);
            }
            else
                usage("unknown cmdline arg '" + arg + "'");
        }

        if (sceneFileName == "")
            usage("no scene file specified");

        // swtiches based on model type, in loaders/loaders.cpp
        Model::SP particles;
        try {
            particles = loadParticleModel(sceneFileName);
            particles->radius = radius;
            std::cout << "Particle radius: " << particles->radius << std::endl;
        }
        catch (std::exception const& e) {
            std::cerr << "Could not create model " << e.what() << std::endl;
            return -1;
        }

        std::cout << "#particles.viewer: creating back-end ..." << std::endl;
        OptixParticles* optixRenderer = nullptr;
        if (method == "allPKD") {
            auto const part = new AllPKDParticles;
            optixRenderer = part;
        }
        else if (method == "noPKD") {
            auto const part = new NoPKDParticles;
            optixRenderer = part;
        }
        else if (method == "treelets") {
            auto const part = new TreeletParticles;
            part->maxTreeletSize = treelet_size;
            optixRenderer = part;
        }
        else if (method == "brute") {
            auto const part = new TreeletParticles;
            part->maxTreeletSize = treelet_size;
            part->bruteForce = true;
            optixRenderer = part;
        }
        else if (method == "treelets-brute") {
            auto const part = new TreeletParticles;
            part->bruteForce = true;
            part->maxTreeletSize = treelet_size;
            optixRenderer = part;
        }
        else
            throw std::runtime_error("unknown acceleration method '" + method + "'");

        optixRenderer->setModel(particles, ext == ".pkd");

        // if (dumpModel) {
        //     dump_model(particles);
        // }

        owl::viewer::GlutWindow::SP window
            = owl::viewer::GlutWindow::prepare(vec2i(cmdline.windowSize.x, cmdline.windowSize.y),
                owl::viewer::GlutWindow::UINT8_RGBA,
                "particlesViewer");
        ModelViewer widget(window, *optixRenderer);
        widget.frameState.samplesPerPixel = spp;
        widget.frameState.c_occ = c_occ;
        widget.frameState.debugOutput = debug;
        widget.frameState.convergenceIterations = convIter;
        widget.frameState.nBudget = nBudget;
        widget.frameState.quant = quant;
        widget.frameState.interp = interp;
        widget.frameState.kernelSize = kernelSize;

        box3f sceneBounds = particles->getBounds();
        widget.enableInspectMode();
        // /* valid range of poi*/sceneBounds,
        //                          /* min distance      */1e-3f,
        //                          /* max distance      */1e6f);
        widget.enableFlyMode();

        widget.setWorldScale(length(sceneBounds.span()));
        if (cmdline.camera.vu != vec3f(0.f)) {
            PING;
            widget.setCameraOrientation(/*origin   */cmdline.camera.vp,
                /*lookat   */cmdline.camera.vi,
                /*up-vector*/cmdline.camera.vu,
                /*fovy(deg)*/70.f);
        }
        else {
            widget.setCameraOrientation(/*origin   */
                sceneBounds.center()
                + vec3f(-.3, .7, 1) * sceneBounds.span(),
                //+ cam_offset,
                /*lookat   */sceneBounds.center(),
                /*up-vector*/vec3f(0, 1, 0),
                //up_vector,
                /*fovy(deg)*/70);
        }
        if (cmdline.windowSize != vec2i(0)) {
            glutReshapeWindow(cmdline.windowSize.x, cmdline.windowSize.y);
        }
        owl::viewer::GlutWindow::run(widget);
    }
} // ::pkd
