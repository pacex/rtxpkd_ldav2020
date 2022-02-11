# occlusion.py

# here, we consider the "tube" extruded from a single screen-space pixel and the particles that are contained within it.
# (x,y) is irrelevant, particles only need a depth coordinate in eye space and a mapping to the voxel their center lives in.
# for simplicity, we do not look at actual voxels, but assume the voxel traversal step is set such as to hit a different voxel each time,
# so our voxels are basically something like a "Bresenham chain" of cells. This makes splatting incorrect for larger particles and oblique rays,
# but that probably does not matter.

import random
import string
import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
import math
from sys import float_info

# parameters
dataset_offset = 1
dataset_length = 25
num_particles = 500
num_rays = 1000
radius = 0.02
gauss_stepping = 0.05
random.seed(42)
num_voxels = 16

voxel_length = dataset_length / num_voxels

def gauss(x: float, sigma: float) -> float:
    mu = 0
    return (0.398942 * pow(2.71828, -(0.5 * (x - mu) * (x - mu)) / (sigma * sigma))) / sigma
gauss_half_width = 1.5
gauss_width = 2 * gauss_half_width
gauss_values = [gauss(x, 0.44) for x in np.arange(-gauss_half_width, gauss_half_width, gauss_stepping)]
gauss_integral = []
sum = 0
for x in gauss_values:
    sum += x
    gauss_integral.append(sum)

# let us normalize the gauss integral
gauss_integral_normalized = [x / sum for x in gauss_integral]

# plt.plot(gauss_values)
# plt.plot(gauss_integral_normalized)
# plt.show()

particles = []
rays = []

def depth_to_voxel(z: float) -> float:
    return (z - dataset_offset) / voxel_length          # do we want to make the first sample hit the center of the first voxel? or what?

def voxel_start(idx: int) -> float:
    return dataset_offset + idx * voxel_length

def voxel_end(idx: int) -> float:
    return dataset_offset + (idx + 1) * voxel_length

class particle:
    z: float
    y: float
    r: float
    voxel_center: int
    voxel_min: int
    voxel_max: int
    def __init__(self, height: float, depth:float, radius: float):
        self.z = depth
        self.y = height
        self.r = radius
        dtv_min = depth_to_voxel(depth - radius)
        dtv_max = depth_to_voxel(depth + radius)
        dtv_center = depth_to_voxel(depth)
        self.voxel_center = int(dtv_center)
        self.voxel_min = max(0, int(dtv_min))
        self.voxel_max = min(num_voxels - 1, int(dtv_max))
        print(self.to_string())

    def to_string(self) -> string:
        return f"particle at ({self.z}, {self.y}), radius {self.r}, in voxel {self.voxel_center}{f' spanning [{self.voxel_min},{self.voxel_max}]' if self.voxel_min != self.voxel_max else ''}"

    def overlap(self, coord: float, start: float, end: float) -> float:
        new_start = max(coord - self.r, start)
        new_end = min(coord + self.r, end)
        overlap = (new_end - new_start) / (2 * self.r)
        print(f"[{start},{end}] -> [{new_start},{new_end}] -> overlap {overlap}, ", end='')
        return overlap
    
    def integral_overlap(self, overlap: float) -> float:
        int_overlap = int(overlap * (len(gauss_integral_normalized) - 1))
        int_overlap = max(0, min(int_overlap, len(gauss_integral_normalized)-1))
        print(f"int_overlap {int_overlap}")
        return int_overlap

    def splat(self, voxel_density):
        for idx in range(self.voxel_min, self.voxel_max + 1):
            print(f"splat: [{self.y}, {self.z}] rad {self.r}:")

            print(f"overlap_h ", end='')
            overlap_h = self.overlap(self.z, voxel_start(idx), voxel_end(idx))
            int_overlap_h = self.integral_overlap(overlap_h)

            print(f"overlap_v ", end='')
            overlap_v = self.overlap(self.y, 0, voxel_length)
            int_overlap_v = self.integral_overlap(overlap_v)
            
            fraction_h = gauss_integral_normalized[int_overlap_h]
            fraction_v = gauss_integral_normalized[int_overlap_v]
            gauss = fraction_h * fraction_v # is that legal and correct?
            print(f"fraction_h {fraction_h}, fraction_v {fraction_v} -> gauss {gauss}")
            voxel_density[idx] += gauss

    def does_intersect(self, ray_y: float) -> bool:
        # thank you parallel rays, I guess
        return True if abs(ray_y - self.y) <= self.r else False

    def intersect(self, ray_y: float) -> float:
        # particle = (z - self.z)^2 + (y - self.y)^2 = self.r^2
        # eye = (ray_y, 0)
        # ray = eye + (0, 1) * t
        # t = -sqrt(-self.y^2 -ray_y^2 + 2 * (self.y * ray_y) + self.r^2) + self.z - 0
        t = -math.sqrt(- self.y ** 2 - ray_y ** 2 + 2 * (self.y * ray_y) + self.r ** 2) + self.z
        return t;


# generate particles
for i in range(num_particles):
    z = random.random() * dataset_length + dataset_offset
    y = random.random() * voxel_length
    particles.append(particle(y, z, radius))

fig, ax = plt.subplots()
circles = [plt.Circle((p.z, p.y), p.r) for p in particles]
coll = matplotlib.collections.PatchCollection(circles, facecolors='black')
ax.add_collection(coll)
rects = [plt.Rectangle((i * voxel_length + dataset_offset, 0), voxel_length, voxel_length, fill=False) for i in range(num_voxels)]
coll = matplotlib.collections.PatchCollection(rects, edgecolors='orange', facecolors='none')
ax.add_collection(coll)
ax.margins(0.01)
plt.xlim(0, dataset_length + dataset_offset)
plt.ylim(- (dataset_length + dataset_offset) / 2, (dataset_length + dataset_offset) / 2)
plt.title("Debug Visualization")
plt.show(block=False)

# generate rays
for i in range(num_rays):
    rays.append(random.random() * voxel_length)

# show center depth histogram
plt.figure()
#plt.hist([x.z for x in particles], density=False, bins = num_voxels)
plt.hist([x.voxel_center for x in particles], density=False, bins = num_voxels)
plt.ylabel('Population')
plt.xlabel('Depth')
plt.title('Particle center depth distribution')
plt.show(block=False)

# splat the particles, to approximate Mohamed's approach
voxel_density = [0 for v in range(num_voxels)]
for p in particles:
    p.splat(voxel_density)

plt.figure()
plt.plot(voxel_density)
plt.ylabel('Density')
plt.xlabel('Depth')
plt.title('Particle density splatted')
plt.show(block=False)

# now let us start raycasting?
print(f"casting {num_rays} rays into {num_particles} particles...")
rays_that_hit = 0
hit_sequence = []
hit_sequence_depth = []
for r in rays:
    first: particle = None
    nearest: particle = None
    hit_depth = float_info.max
    for p in particles:
        if p.does_intersect(r):
            t = p.intersect(r)
            if first == None:
                first = p
                nearest = p
            if hit_depth > t:
                nearest = p
                hit_depth = t
    if first != None:
        rays_that_hit += 1
        hit_sequence.append(nearest)
        hit_sequence_depth.append(hit_depth)
        print(f"ray {r} hit {nearest.to_string()} at depth {hit_depth}")

print(f"out of {num_rays} rays, {rays_that_hit} hit something.")

plt.figure()
#plt.plot([p.z for p in hit_sequence])
plt.plot([d for d in hit_sequence_depth])
plt.xlabel('ray #')
plt.ylabel('nearest hit')
plt.title('Depth sequence of nearest hits')
plt.show()

# what is the deepest depth we found with brute force? what would be the result? like hit/miss ratio i.e. saturation?

# other approaches