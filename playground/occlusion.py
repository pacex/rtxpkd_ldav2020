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


def simulate(num_particles: int = 2000, num_rays: int = 1000, radius: float = 0.08, seed: int = 42, particle_voxel_offset: float = 0.0, C_occ: float = 0.95, use_splatting: bool = True, plot: bool = True, log: bool = True):

    # parameters
    dataset_offset = 1
    dataset_length = 25
    #num_particles = 1000
    #num_rays = 1000
    #radius = 0.02
    gauss_stepping = 0.05
    random.seed(seed)
    num_voxels = 16
    particle_offset = particle_voxel_offset * (dataset_length / num_voxels) # How many voxels behind dataset offset do particles actually start appearing?
    use_vanilla_raycasting = True
    use_probabilistic_culling = True
    #use_splatting = True
    N_budget = 25

    voxel_length = dataset_length / num_voxels

    # Accelerate splatting: Sample 1D gauss kernel, then integrate the samples.
    # Then we could find the splat contribution by just computing the overlap of the splat and the voxel
    # and looking that up in the pre-integrated array. Since it is symmetric, we do not even care
    # about the "direction" of the overlap. right?
    def gauss(x: float, sigma: float) -> float:
        mu = 0
        return (0.398942 * pow(2.71828, -(0.5 * (x - mu) * (x - mu)) / (sigma * sigma))) / sigma

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
            if log:
                print(self.to_string())

        def to_string(self) -> string:
            return f"particle at ({self.z}, {self.y}), radius {self.r}, in voxel {self.voxel_center}{f' spanning [{self.voxel_min},{self.voxel_max}]' if self.voxel_min != self.voxel_max else ''}"

        def overlap(self, coord: float, start: float, end: float) -> float:
            new_start = max(coord - self.r, start)
            new_end = min(coord + self.r, end)
            overlap = (new_end - new_start) / (2 * self.r)
            if log:
                print(f"[{start},{end}] -> [{new_start},{new_end}] -> overlap {overlap}, ", end='')
            return overlap

        def integral_overlap(self, overlap: float) -> float:
            int_overlap = int(overlap * (len(gauss_integral_normalized) - 1))
            int_overlap = max(0, min(int_overlap, len(gauss_integral_normalized)-1))
            if log:
                print(f"int_overlap {int_overlap}")
            return int_overlap

        # along the lines of separable kernels, can we just compute the two 1D
        # overlaps and then multiply the respective "portions" of the pre-integration?
        def splat(self, voxel_density):

            if not use_splatting:
                voxel_density[self.voxel_center] += 1
                return

            for idx in range(self.voxel_min, self.voxel_max + 1):
                if log:
                    print(f"splat: [{self.y}, {self.z}] rad {self.r}:")

                if log:
                    print(f"overlap_h ", end='')
                overlap_h = self.overlap(self.z, voxel_start(idx), voxel_end(idx))
                int_overlap_h = self.integral_overlap(overlap_h)

                if log:
                    print(f"overlap_v ", end='')
                overlap_v = self.overlap(self.y, 0, voxel_length)
                int_overlap_v = self.integral_overlap(overlap_v)

                fraction_h = gauss_integral_normalized[int_overlap_h]
                fraction_v = gauss_integral_normalized[int_overlap_v]
                gauss = fraction_h * fraction_v # is that legal and correct?
                if log:
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


    # render the dataset, haha
    def render_dataset(parts, title):
        if not plot:
            return

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




    # functions needed for probabilistic culling

    def accumulate_confidence(C: float, a: float, delta_Enk: float):
        inv_c = 1.0 - C
        inv_c *= pow(1-a, delta_Enk)
        return 1.0 - inv_c

    def expected_number_of_unique_particles(N: float, k: int):
        # Eq. (5)
        M = 1.0 - 1.0 / N
        return (1.0 - pow(M, k)) / (1.0 - M)


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

    def particle_count_between_depths(d1: float, d2: float):
        # This is the D(d1,d2) function from Mohamed's paper,
        # i.e. how many particles do we estimate to be in the pixel tube clipped to d1 and d2?

        if d1 >= d2 or d2 <= voxel_start(0) or d1 >= voxel_end(num_voxels-1):
            return 0.0

        psum = 0.0
        v1 = max(0, min(num_voxels - 1, math.floor(depth_to_voxel(d1))))  # the voxel that d1 lives in
        v2 = max(0, min(num_voxels - 1, math.floor(depth_to_voxel(d2))))  # the voxel that d2 lives in

        d1 = max(voxel_start(v1), d1)
        d2 = min(voxel_end(v2), d2)

        # The return value of this function is only dependent on the voxels d1 and d2 live in.
        # If e.g. d1 varies within one voxel, the same value will be returned.
        # Is weighing samples from v1 and v2 by (voxel_end(v1) - d1) / voxel_length and
        # (d2 - voxel_start(v2) / voxel_length) respectively and option?

        if v1 == v2:
            return voxel_density[v1] * ((d2 - d1) / voxel_length)

        psum += voxel_density[v1] * ((voxel_end(v1) - d1) / voxel_length)

        # no interpolation required as sample points lie on voxel centers and voxels match pixel tube in height
        # also voxels are fully inside the pixel tube => weight = 1
        for i in range(v1 + 1, v2):
            psum += voxel_density[i]

        psum += voxel_density[v2] * ((d2 - voxel_start(v2)) / voxel_length)

        return psum

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

    # generate particles
    for i in range(num_particles):
        z = random.random() * dataset_length + dataset_offset
        if z < dataset_offset + particle_offset:
            continue;
        y = random.random() * voxel_length
        particles.append(particle(y, z, radius))

    render_dataset(particles, "Whole data")

    # generate rays
    for i in range(num_rays):
        rays.append(random.random() * voxel_length)

    # show center depth histogram
    if plot:
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
    if plot:
        plt.figure()
        plt.plot(voxel_density)
        plt.ylabel('Density')
        plt.xlabel('Depth')
        plt.title('Particle density splatted')
        plt.show(block=False)

    # show estimated particle counts behind certain depths
    voxel_density_integrated = [particle_count_between_depths(voxel_start(v), float_info.max) for v in range(num_voxels)]
    if plot:
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
        if log:
            print(f"casting {num_rays} rays into {num_particles} particles (vanilla)...")
        rays_that_hit_vanilla = 0
        hit_sequence = []
        hit_sequence_depth_vanilla = []
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
                rays_that_hit_vanilla += 1
                hit_sequence.append(nearest)
                hit_sequence_depth_vanilla.append(hit_depth)
                if log:
                    print(f"ray {r} hit {nearest.to_string()} at depth {hit_depth}")
            else:
                hit_sequence_depth_vanilla.append(np.nan)
        if log:
            print(f"out of {num_rays} rays, {rays_that_hit_vanilla} hit something.")
        useful_particles_vanilla = list(set([p for p in hit_sequence]))
        if log:
            print(f"but (out of {num_particles} total particles) we only actually hit {len(useful_particles_vanilla)} different particles.")

        # what is the dataset "front"
        render_dataset(useful_particles_vanilla, "hit particles (normal raycasting)")

        # what kind of depths did we hit, and when?
        if plot:
            plt.figure()
            #plt.plot([p.z for p in hit_sequence])
            plt.plot([d for d in hit_sequence_depth_vanilla], 'x')
            plt.xlabel('ray #')
            plt.ylabel('nearest hit')
            depth_seq_min = min(hit_sequence_depth_vanilla)
            depth_seq_max = max(hit_sequence_depth_vanilla)
            plt.gca().set_ylim([depth_seq_min, depth_seq_max])
            plt.title('Depth sequence of nearest hits (normal raycasting)')
            plt.show(block=False)

    # ===========================================
    #     RAY CASTING (PROBABILISTIC CULLING)
    # ===========================================
    if use_probabilistic_culling:
        if log:
            print(f"casting {num_rays} rays into {num_particles} particles (using probabilistic culling)...")
        rays_that_hit_probabilistic = 0
        rays_that_miss_probilistic = 0
        rays_culled_probabilistic = 0
        hit_sequence = []
        hit_sequence_depth_probabilistic = []

        t_max = float_info.max  # t_max of rays cast
        t_cull = float_info.max  # to keep names uniform, we call d_cull t_cull in this prototype
        C_cull = 0.0
        k_cull = 1
        t_accum = float_info.max
        C_accum = 0.0
        k_accum = 1

        Enk_previous_cull = 0.0
        Enk_previous_accum = 0.0

        a_sample = (2 * radius) / voxel_length

        ray_index = 0
        for r in rays:
            first: particle = None
            nearest: particle = None
            hit_depth = float_info.max

            # This is what we do instead of moving meshlets behind a certain depth into the occluded class:
            # We take a per pixel decision by clipping rays cast at t_max = t_cull if the culling confidence in sufficient.
            if C_cull > C_occ:
                t_max = t_cull

            # ray cast
            too_deep = False
            for p in particles:
                if p.does_intersect(r):
                    t = p.intersect(r)

                    # clip ray at t_max
                    if t > t_max:
                        too_deep = True
                        continue

                    if first == None:
                        first = p
                        nearest = p
                    if hit_depth > t:
                        nearest = p
                        hit_depth = t

            # handle hit
            if first == None:
                if log:
                    print(f"{ray_index} | ray {r} hit nothing{': too deep' if too_deep else ''}")
                hit_sequence_depth_probabilistic.append(np.nan)
                if too_deep:
                    rays_culled_probabilistic += 1
                else:
                    rays_that_miss_probilistic += 1
            else:
                rays_that_hit_probabilistic += 1
                hit_sequence.append(nearest)
                hit_sequence_depth_probabilistic.append(hit_depth)
                if log:
                    print(f"{ray_index} | ray {r} hit {nearest.to_string()} at depth {hit_depth}")

                B_tmin = particle_count_between_depths(0.0, t_max)
                B_tsample = particle_count_between_depths(hit_depth, t_max)
                B_tcull = particle_count_between_depths(t_cull, t_max)
                B_taccum = particle_count_between_depths(t_accum, t_max)



                # Estimate of #particles we can hit with a ray cast out of this pixel's footprint
                N = max(1.0, B_tmin)

                #print(f"{rays_that_hit_probabilistic} | N= {N}, t_max= {t_max}, t_sample= {hit_depth}, B_tsample= {B_tsample}, t_cull= {t_cull}, C_cull= {C_cull}, B_tcull= {B_tcull}, t_accum= {t_accum}, C_accum= {C_accum}, B_taccum= {B_taccum}")
                if log:
                    print("%i | N= %.4f, t_max= %.2e, t_sample= %.4f, B_tsample= %.4f, t_cull= %.2e, C_cull= %.4f, B_tcull= %.4f, t_accum= %.2e, C_accum = %.4f, B_taccum= %.4f"
                          % (ray_index,N,t_max,hit_depth,B_tsample, t_cull, C_cull, B_tcull, t_accum, C_accum, B_taccum))

                # Algorithm 1:
                if hit_depth <= t_cull:
                    Enk_cull = expected_number_of_unique_particles(max(1.0, B_tmin - B_tcull), k_cull)
                    delta_Enk_cull = Enk_cull - Enk_previous_cull
                    C_cull = accumulate_confidence(C_cull, a_sample, delta_Enk_cull)
                    k_cull += 1
                    Enk_previous_cull = Enk_cull

                if hit_depth <= t_accum:
                    accProb = acceptance_probability(C_accum, B_taccum, B_tsample, B_tmin, a_sample)
                    if random.random() <= accProb and hit_depth < t_accum:
                        t_accum = voxel_end(depth_to_voxel(hit_depth))
                        C_accum = a_sample
                        Enk_previous_accum = 0.0
                        k_accum = 1
                    else:
                        Enk_accum = expected_number_of_unique_particles(max(1.0, B_tmin - B_taccum), k_accum)
                        delta_Enk_accum = Enk_accum - Enk_previous_accum
                        C_accum = accumulate_confidence(C_accum, a_sample, delta_Enk_accum)
                        k_accum += 1
                        Enk_previous_accum = Enk_accum

                if B_taccum * C_accum > B_tcull * C_cull:
                    C_cull = C_accum
                    t_cull = t_accum
                    k_cull = k_accum
                    Enk_previous_cull = Enk_previous_accum

            ray_index += 1

        if log:
            print(f"out of {num_rays} rays, {rays_that_hit_probabilistic} hit something.")
        useful_particles_probabilistic = list(set([p for p in hit_sequence]))
        if log:
            print(f"but (out of {num_particles} total particles) we only actually hit {len(useful_particles_probabilistic)} different particles.")

        # what is the dataset "front"
        render_dataset(useful_particles_probabilistic, "hit particles (probabilistic)")

        # what kind of depths did we hit, and when?
        if plot:
            plt.figure()
            #plt.plot([p.z for p in hit_sequence])
            plt.plot([d for d in hit_sequence_depth_probabilistic], 'o')
            plt.xlabel('ray #')
            plt.ylabel('nearest hit')
            if use_vanilla_raycasting:
                plt.gca().set_ylim([depth_seq_min, depth_seq_max])
            plt.title('Depth sequence of nearest hits (probabilistic)')
            plt.show(block=False)

    if plot:
        plt.figure()
        plt.xlabel('ray #')
        plt.ylabel('nearest hit')
        plt.plot([d for d in hit_sequence_depth_probabilistic], 'o', label=f'probabilistic: {len(useful_particles_probabilistic)} particles, {rays_that_hit_probabilistic} hits, {rays_culled_probabilistic} (wrongly) culled, {rays_that_miss_probilistic} misses')
        plt.plot([d for d in hit_sequence_depth_vanilla], 'x', label=f'vanilla: {len(useful_particles_vanilla)} particles, {rays_that_hit_vanilla} hits, {num_rays - rays_that_hit_vanilla} misses')
        plt.gca().legend()
        plt.title('combined depth sequence')
        plt.show()

    return rays_culled_probabilistic


simulate()