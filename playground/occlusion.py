# occlusion.py

# here, we consider the "tube" extruded from a single screen-space pixel and the particles that are contained within it.
# x is irrelevant, particles only need a y and depth coordinate in eye space and a mapping to the voxel their center lives in.
# for simplicity, we do look at rays that are aligned with the voxels "chain", so we do not need actual raycasting

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
use_vanilla_raycasting = True
use_probabilistic_culling = True
N_budget = 25

voxel_length = dataset_length / num_voxels

# Accelerate splatting: Sample 1D gauss kernel, then integrate the samples.
# Then we could find the splat contribution by just computing the overlap of the splat and the voxel
# and looking that up in the pre-integrated array. Since it is symmetric, we do not even care
# about the "direction" of the overlap. right?
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

# normalize the gauss integral, later we scale it to the proper "radius"
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

    # along the lines of separable kernels, can we just compute the two 1D
    # overlaps and then multiply the respective "portions" of the pre-integration?
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

# render the dataset, haha
def render_dataset(parts, title):
    fig, ax = plt.subplots()
    circles = [plt.Circle((p.z, p.y), p.r) for p in parts]
    coll = matplotlib.collections.PatchCollection(circles, facecolors='black')
    ax.add_collection(coll)
    rects = [plt.Rectangle((i * voxel_length + dataset_offset, 0), voxel_length, voxel_length, fill=False) for i in range(num_voxels)]
    coll = matplotlib.collections.PatchCollection(rects, edgecolors='orange', facecolors='none')
    ax.add_collection(coll)
    ax.margins(0.01)
    plt.xlim(0, dataset_length + dataset_offset)
    plt.ylim(- (dataset_length + dataset_offset) / 2, (dataset_length + dataset_offset) / 2)
    plt.title(title)
    plt.show(block=False)

render_dataset(particles, "Whole data")


# functions needed for probabilistic culling

def accumulate_confidence(N: float, a: float, k: int):
    M = 1.0 - (1.0 / N)
    Enk = (1.0 - pow(M, k)) / (1.0 - M)  # Eq. (5)
    return 1.0 - pow(1.0 - a, Enk)  # Eq. (7)

def acceptance_probability(C_accum: float, B_taccum: float, B_tsample: float, B_tmin: float, a: float):
    # Eq. (15)
    n = math.log(1.0 - (B_taccum / B_tsample) * C_accum, 1.0 - a)
    # As Eq. (15) is an inequality that sets the lower bound for minimum unique particles n,
    # we set n as the next larger integer
    if math.fmod(n, 1.0) == 0.0:
        n += 1
    else:
        n = math.ceil(n)

    # D(t_min, t_sample) = B(t_min) - B(t_sample) = D(t_min, t_max) - D(t_sample, t_max)
    D_tmin_ts = B_tmin - B_tsample

    # Eq. (20)
    if n > 0.98 * D_tmin_ts:
        return 0.0

    # Eq. (18)
    M = 1.0 - (1.0 / D_tmin_ts)

    # Eq. (19)
    k = math.ceil(math.log(1.0 - (n / D_tmin_ts), M))

    # Eq. (22)
    p = D_tmin_ts / B_tmin

    # Eq. (30) [App. A]
    mu = N_budget * p
    sigma = N_budget * p * (1-p)

    return 1.0 - cdf_normal((k + 0.5 - mu) / math.sqrt(sigma))

def cdf_normal(x):
    # In the real program, this would be a lookup of precomputed values
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def particle_count_behind_depth(d1: float, d2: float):
    # This is the D(d1,d2) function from Mohamed's paper,
    # i.e. how many particles do we estimate to be in the pixel tube clipped to d1 and d2?

    if d1 >= d2:
        return 0.0

    psum = 0.0
    v1 = max(0, min(num_voxels - 1, math.floor(depth_to_voxel(d1))))  # the voxel that d1 lives in
    v2 = max(0, min(num_voxels - 1, math.floor(depth_to_voxel(d2))))  # the voxel that d2 lives in

    # The return value of this function is only dependent on the voxels d1 and d2 live in.
    # If e.g. d1 varies within one voxel, the same value will be returned.
    # Is weighing samples from v1 and v2 by (voxel_end(v1) - d1) / voxel_length and
    # (d2 - voxel_start(v2) / voxel_length) respectively and option?

    # no interpolation required as sample points lie on voxel centers and voxels match pixel tube in height
    # also voxels are fully inside the pixel tube => weight = 1
    for i in range(v1, v2 + 1):
        psum += voxel_density[i]

    return psum



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

# show splatting result
plt.figure()
plt.plot(voxel_density)
plt.ylabel('Density')
plt.xlabel('Depth')
plt.title('Particle density splatted')
plt.show(block=False)

# show estimated particle counts behind certain depths
voxel_density_integrated = [particle_count_behind_depth(voxel_start(v), float_info.max) for v in range(num_voxels)]
plt.figure()
plt.plot(voxel_density_integrated)
plt.ylabel('Density')
plt.xlabel('Depth')
plt.title('Particle density integrated behind depth')
plt.show(block=False)


# =============================
#     RAY CASTING (VANILLA)
# =============================
if use_vanilla_raycasting:
    print(f"casting {num_rays} rays into {num_particles} particles (vanilla)...")
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
    useful_particles = list(set([p for p in hit_sequence]))
    print(f"but (out of {num_particles} total particles) we only actually hit {len(useful_particles)} different particles.")

    # what is the dataset "front"
    render_dataset(useful_particles, "hit particles (normal raycasting)")

    # what kind of depths did we hit, and when?
    plt.figure()
    #plt.plot([p.z for p in hit_sequence])
    plt.plot([d for d in hit_sequence_depth])
    plt.xlabel('ray #')
    plt.ylabel('nearest hit')
    depth_seq_min = min(hit_sequence_depth)
    depth_seq_max = max(hit_sequence_depth)
    plt.gca().set_ylim([depth_seq_min, depth_seq_max])
    plt.title('Depth sequence of nearest hits (normal raycasting)')
    plt.show(block=False)

# ===========================================
#     RAY CASTING (PROBABILISTIC CULLING)
# ===========================================
if use_probabilistic_culling:
    print(f"casting {num_rays} rays into {num_particles} particles (using probabilistic culling)...")
    rays_that_hit = 0
    hit_sequence = []
    hit_sequence_depth = []

    t_max = float_info.max  # t_max of rays cast
    t_cull = float_info.max  # to keep names uniform, we call d_cull t_cull in this prototype
    C_cull = 0.0
    k_cull = 1
    t_accum = float_info.max
    C_accum = 0.0
    k_accum = 1

    a_sample = (2 * radius) / voxel_length


    for r in rays:
        first: particle = None
        nearest: particle = None
        hit_depth = float_info.max

        # This is what we do instead of moving meshlets behind a certain depth into the occluded class:
        # We take a per pixel decision by clipping rays cast at t_max = t_cull if the culling confidence in sufficient.
        if C_cull > 0.95:
            t_max = t_cull

        # ray cast
        for p in particles:
            if p.does_intersect(r):
                t = p.intersect(r)

                # clip ray at t_max
                if t > t_max:
                    continue

                if first == None:
                    first = p
                    nearest = p
                if hit_depth > t:
                    nearest = p
                    hit_depth = t

        # handle hit
        if first != None:
            rays_that_hit += 1
            hit_sequence.append(nearest)
            hit_sequence_depth.append(hit_depth)
            print(f"ray {r} hit {nearest.to_string()} at depth {hit_depth}")

            B_tmin = particle_count_behind_depth(0.0, t_max)
            B_tsample = particle_count_behind_depth(hit_depth, t_max)
            B_tcull = particle_count_behind_depth(t_cull, t_max)
            B_taccum = particle_count_behind_depth(t_accum, t_max)

            # Estimate of #particles we can hit with a ray cast out of this pixel's footprint
            N = max(1.0, B_tmin)

            #print(f"{rays_that_hit} | N= {N}, t_max= {t_max}, t_sample= {hit_depth}, B_tsample= {B_tsample}, t_cull= {t_cull}, C_cull= {C_cull}, B_tcull= {B_tcull}, t_accum= {t_accum}, C_accum= {C_accum}, B_taccum= {B_taccum}")
            print("%i | N= %.4f, t_max= %.2e, t_sample= %.4f, B_tsample= %.4f, t_cull= %.4f, C_cull= %.4f, B_tcull= %.4f, t_accum= %.4f, C_accum = %.4f, B_taccum= %.4f"
                  % (rays_that_hit,N,t_max,hit_depth,B_tsample, t_cull, C_cull, B_tcull, t_accum, C_accum, B_taccum))

            # Algorithm 1:
            if hit_depth <= t_cull:
                C_cull = accumulate_confidence(N, a_sample, k_cull)
                k_cull += 1

            if hit_depth <= t_accum:
                accProb = acceptance_probability(C_accum, B_taccum, B_tsample, B_tmin, a_sample)
                if random.random() <= accProb and hit_depth < t_accum:
                    t_accum = hit_depth
                    C_accum = a_sample
                    k_accum = 1
                else:
                    C_accum = accumulate_confidence(N, a_sample, k_accum)
                    k_accum += 1

            if B_taccum * C_accum > B_tcull * C_cull:
                C_cull = C_accum
                t_cull = t_accum
                k_cull = k_accum

    print(f"out of {num_rays} rays, {rays_that_hit} hit something.")
    useful_particles = list(set([p for p in hit_sequence]))
    print(
        f"but (out of {num_particles} total particles) we only actually hit {len(useful_particles)} different particles.")

    # what is the dataset "front"
    render_dataset(useful_particles, "hit particles (probabilistic)")

    # what kind of depths did we hit, and when?
    plt.figure()
    #plt.plot([p.z for p in hit_sequence])
    plt.plot([d for d in hit_sequence_depth])
    plt.xlabel('ray #')
    plt.ylabel('nearest hit')
    if use_vanilla_raycasting:
        plt.gca().set_ylim([depth_seq_min, depth_seq_max])
    plt.title('Depth sequence of nearest hits (probabilistic)')
    plt.show(block=False)

plt.figure()
plt.title('just for blocking the program flow')
plt.show()