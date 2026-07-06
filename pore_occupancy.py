#!/usr/bin/env python3
"""
pore_occupancy.py  -  quantitative pore-lipid occupancy metric

Idea:
  1. Normalize every map to a consistent, invariant feature near the pore with a reference mask.
  2. Measure the density inside a defined "pore" volume.
  3. Anchor to two controls: a lipidated map = 1.0, a delipidated map = 0.0.
  4. Report every other map on that 0-1 scale.

REQUIREMENTS:
  - pip install mrcfile numpy --break-system-packages
  - ALL maps and BOTH masks must be on the SAME grid.
"""
import mrcfile
import numpy as np

#Paths to pore mask and reference mask
BASE = "C:/Users/rjw89/OneDrive/Desktop/BrohawnLab"
pore_mask_file = BASE + "/4)PoreLipidStuff/PoreMap_258.mrc"   # 258^3 (resample PoreMap.mrc onGrid a 258 map in ChimeraX)
ref_mask_file   = BASE + "/4)PoreLipidStuff/StableReigonMap.mrc"   
#paths to control structures that make the scale
control_lipidated   = BASE + "/1)PDBstrucutres/OblongMap.mrc"     #1.0 control
control_delipidated = BASE + "/1)PDBstrucutres/OblongT48DM.mrc"   #0.0 control

# --- volumes grouped by 3D-class job
P = BASE + "/4)PoreLipidStuff"
jobs = {
    "MT6 537": [("538", P+"/MT63DClass(537)/538.mrc"), ("539", P+"/MT63DClass(537)/539.mrc"),
                ("540", P+"/MT63DClass(537)/540.mrc"), ("541", P+"/MT63DClass(537)/541.mrc")],
    "MT6 560": [("591", P+"/MT63DClass(560)/591.mrc"), ("592", P+"/MT63DClass(560)/592.mrc"),
                ("593", P+"/MT63DClass(560)/593.mrc"), ("594", P+"/MT63DClass(560)/594.mrc")],
    "MT6 582": [("588", P+"/MT63DClass(582)/588.mrc"), ("589", P+"/MT63DClass(582)/589.mrc"),
                ("590", P+"/MT63DClass(582)/590.mrc")],
    "RW1 622": [("638", P+"/RW13DClass(622)/638.mrc"), ("649", P+"/RW13DClass(622)/649.mrc"),
                ("651", P+"/RW13DClass(622)/651.mrc")],
    "RW1 660": [("668", P+"/RW13DClass(660)/668Resamp.mrc"), ("734", P+"/RW13DClass(660)/734Resamp.mrc"),
                ("735", P+"/RW13DClass(660)/735Resamp.mrc")],
    "RW1 753": [("758", P+"/RW13DClass(753)/758.mrc"), ("759", P+"/RW13DClass(753)/759.mrc"),
                ("760", P+"/RW13DClass(753)/760.mrc")],
}
#dictionary with key as each 3D class and Values as the paths

def load(f):
    with mrcfile.open(f, permissive=True) as m:
        return np.asarray(m.data, dtype=np.float64)

pore = load(pore_mask_file)
ref  = load(ref_mask_file)

def occupancy(map_file): # ---- pore occupancy = mean normalized density inside the pore mask
    d = load(map_file)
    if d.shape != pore.shape: #make sure all on same grid
        raise ValueError(
            f"{map_file} shape {d.shape} != mask shape {pore.shape};")
    #Numpy Basics -> * multiplies voxel values at each point in space   ---   sum adds the voxel density values
    ref_mean = (d * ref).sum() / ref.sum()  #how much the map occupies the stable reference (e.g. 0.6 would mean 60 percent occupied)
    if ref_mean <= 0:                       # guard against odd scaling bc negative density values
        ref_mean = abs(ref_mean) + 1e-9
    d_norm = d / ref_mean        #normilize the density by dividing by the maps occupacy of the stable reigon
    return (d_norm * pore).sum() / pore.sum()   #normalized density in the pore / pore map density

occ_lip   = occupancy(control_lipidated)
occ_delip = occupancy(control_delipidated)
span = occ_lip - occ_delip

print(f"{'CONTROL lipidated  (=1)':32s} raw {occ_lip:8.4f}")
print(f"{'CONTROL delipidated (=0)':32s} raw {occ_delip:8.4f}")
print("-" * 60)
for jobname, classes in jobs.items():
    print(f"\n=== {jobname} ===   (% = pore-density difference on David's lipidated->delipidated scale)")
    rows = sorted([(cls, occupancy(f)) for cls, f in classes], key=lambda r: r[1], reverse=True)
    top_cls, top_v = rows[0]
    for cls, v in rows:
        if cls == top_cls:
            print(f"   class {cls:6s} raw {v:8.4f}   (highest lipid, reference)")
        else:
            d_pct = (v - top_v) / span * 100.0 if span != 0 else float("nan")
            print(f"   class {cls:6s} raw {v:8.4f}   {d_pct:+6.1f}% vs {top_cls}")
    spread = (rows[0][1] - rows[-1][1]) / span * 100.0 if span != 0 else float("nan")
    print(f"   within-job spread (highest - lowest): {spread:.1f}% of the David lipid difference")

print("\nInterpretation: normalized ~1 = pore lipid present (like lipidated control),")
print("~0 = pore lipid absent (like delipidated control). Values are an estimate;")
print("report alongside HOLE, difference maps, particle counts, and local resolution.")
