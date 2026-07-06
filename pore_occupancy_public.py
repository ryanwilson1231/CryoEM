#!/usr/bin/env python3
"""
region_occupancy.py
Quantify the density inside a masked region of a cryo-EM map (e.g. a lipid in a
channel pore), normalized so maps of different scale can be compared, and placed
on a 0-1 scale defined by two control maps.

Method
------
For each map:
    occupancy = mean(density in REGION mask) / mean(density in REFERENCE mask)

  - REGION mask  : where you measure (e.g. a tube over the conduction pathway).
  - REFERENCE mask: an invariant, well-ordered part of the structure. Dividing by
    its mean density cancels any global scale factor between maps, so occupancy is
    scale-invariant. (Note: this corrects a multiplicative scale, not an additive
    baseline offset -- for cross-dataset comparisons an affine scale+offset fit
    over the reference region is more robust.)

Two control maps then define the scale: one with the feature present (= 1.0) and
one with it absent (= 0.0). Every other map is reported on that scale.

Requirements
------------
    pip install mrcfile numpy

All maps and both masks MUST be on the same voxel grid (same box / pixel / shape).
If a mask is on a different grid, resample it first, e.g. in ChimeraX:
    vop resample #mask onGrid #a_map
    save resampled_mask.mrc model #<result>
"""

import mrcfile
import numpy as np

# ----------------------------- CONFIG: edit these -----------------------------
region_mask_file = "region_mask.mrc"       # where density is measured
ref_mask_file    = "reference_mask.mrc"    # invariant region used to normalize

control_full  = "control_present.mrc"      # feature present -> occupancy = 1.0
control_empty = "control_absent.mrc"       # feature absent  -> occupancy = 0.0

maps = {                                   # maps to score:  label -> path
    "map_1": "map_1.mrc",
    "map_2": "map_2.mrc",
}
# ------------------------------------------------------------------------------


def load(path):
    """Load an .mrc file as a float64 NumPy array (a 3D grid of density values)."""
    with mrcfile.open(path, permissive=True) as m:
        return np.asarray(m.data, dtype=np.float64)


region = load(region_mask_file)   # soft mask (weights >= 0) over the region of interest
ref    = load(ref_mask_file)      # soft mask over the invariant reference region


def occupancy(map_path):
    """Normalized mean density inside the region mask for one map."""
    d = load(map_path)
    if d.shape != region.shape:
        raise ValueError(
            f"{map_path} shape {d.shape} != mask shape {region.shape}; "
            "resample all maps and masks to a common grid first."
        )
    # normalize: scale the map so the mean density in the reference region == 1
    ref_mean = (d * ref).sum() / ref.sum()
    if ref_mean <= 0:                      # reference should sit on positive protein density
        ref_mean = abs(ref_mean) + 1e-9    # guard against divide-by-zero / sign flip
    d_norm = d / ref_mean
    # measure: mask-weighted mean density inside the region
    return (d_norm * region).sum() / region.sum()


# define the 0-1 scale from the two controls
occ_full  = occupancy(control_full)
occ_empty = occupancy(control_empty)
span = occ_full - occ_empty

print(f"control (present, =1): {occ_full:.4f}")
print(f"control (absent,  =0): {occ_empty:.4f}    span = {span:.4f}")
print("-" * 52)
for label, path in maps.items():
    v = occupancy(path)
    norm = (v - occ_empty) / span if span != 0 else float("nan")
    print(f"{label:20s} raw {v:.4f}   normalized {norm:6.2f}")

print("\nnormalized ~1 = feature present (like the 'present' control),")
print("           ~0 = feature absent  (like the 'absent'  control).")
print("This is a relative estimate; report alongside difference maps and,")
print("for cryo-EM, resolution / particle-count matched comparisons.")
