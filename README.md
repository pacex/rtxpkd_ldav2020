# RTX-PKD - "PKD Trees in Optix/RTX"

This repository shows the experimental code used to obtain the OptiX-related measurements in the paper:  
*"Spatial Partitioning Strategies for Memory-Efficient Ray Tracing of Particles"*, LDAV 2020, to appear  
This application has been used on Windows 10.
It was modified and extended by Pascal Walloner to evaluate probabilistic occlusion culling with P-k-d-based ray tracers
as part of the Bachelor's thesis *"Acceleration of P-k-d tree traversal using probabilistic occlusion"*.

## Building

This application requires CUDA 10.1+, OptiX 7+, and TBB.  
It depends on OWL. Checkout tag *ldav_2020_sub* from the fork <https://github.com/UniStuttgart-VISUS/owl>.  
Original repo <https://github.com/owl-project/owl>.  
The folder structure has to be:  
___ some folder name  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ owl  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ rtxpkd  
for *rtxpkd* to find OWL and to use it in subsequent build steps.

## Running

Run *rtxParticleViewer* [options] &lt;filename&gt;  
The application determines from the file extension which loader to use.
Accepted file types are: p4, xyz, random, pkd, and mmpld.
For mmpld files, the *USE_MMPLD* option has to be ON in the CMake configuration.
*PKD* files are dumps of datasets after pkd construction.
*Random* is a type indicating that a random dataset has to be created by the application.
The syntax is: &lt;num&gt;_&lt;radius&gt;.random.
*Num* determines the number of particles to generate.
The value of *radius* is currently overriden by the default value of the *--radius* argument.  

Subset of arguments accepted by the application:

| Option                    | Description                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------- |
| --allpkd                  | renders scene with a PkD tree (the only supported AS with prob. culling)                |
| --nopkd                   | renders scene with a BVH                                                                |
| --treelets                | renders scene with the nested hierarchy using PkD treelets (combine this with max-size) |
| --treelets-brute          | renders scene with the nested hierarchy using flat lists (combine this with max-size)   |
| --max-size &lt;size&gt;   | determine the maximal size of treelets in the nested hierarchy                          |
| --rec-depth &lt;depth&gt; | recursion depth of the ray tracer (only rec-depth 0 supported with prob. culling)       |
| --radius &lt;radius&gt;   | set radius of the rendered spheres                                                      |
|                           |                                                                                         |
| --c_occ &lt;confidence&gt;| set confidence threshold                                                                |
| --voxel-count &lt;n&gt;   | set density grid resolution                                                             |
| --conv-iter &lt;count&gt; | set convergence iteration count                                                         |
| --quant                   | enable depth quantisation                                                               |
| --interp                  | enable density grid interpolation                                                       |
| --kernel-size &lt;n&gt;   | set neighborhood kernel size (2n+1 x 2n+1)                                              |
