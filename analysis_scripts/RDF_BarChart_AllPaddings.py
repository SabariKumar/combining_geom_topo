import os
os.environ['OMP_NUM_THREADS'] = "64"   # set before NumPy

import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ─── Plot styling ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "Arial",
    "font.size": 18,
    "svg.fonttype": "none"
})

def get_final_derivative_peak_radius(topology, trajectory):
    # Load trajectory
    u = mda.Universe(topology, trajectory)

    prot_res = u.select_atoms("protein")
    water    = u.select_atoms("resname HOH")

    # Compute RDF
    irdf = rdf.InterRDF(
        prot_res, water,
        nbins=75,
        range=(0.0, 15.0)   # Å
    )
    irdf.run()

    r   = irdf.results.bins
    g_r = irdf.results.rdf
    dg  = np.gradient(g_r, r)

    # Find peaks in derivative
    peaks, properties = find_peaks(dg, prominence=0.01)

    if len(peaks) == 0:
        raise ValueError("No meaningful peaks found in derivative curve.")

    final_peak_index = peaks[-1]
    final_peak_radius = r[final_peak_index]

    return final_peak_radius, r, g_r, dg, peaks

# ─── Run for all three simulations ───────────────────────────────────────────
peak_1nm, r1, g1, dg1, peaks1 = get_final_derivative_peak_radius(
    "100ns_CA1_final_positions_fixed.pdb",
    "100ns_CA1_centered.dcd"
)

peak_2nm, r2, g2, dg2, peaks2 = get_final_derivative_peak_radius(
    "20260331_CA1_100ns_final_positions_2nm_try4_fixed.pdb",
    "20260331_CA1_100ns_trajectory_2nm_try4_centered.dcd"
)

peak_3nm, r3, g3, dg3, peaks3 = get_final_derivative_peak_radius(
    "20250331_CA1_100ns_final_positions_3nm_fixed.pdb",
    "20250331_CA1_100ns_trajectory_3nm_centered.dcd"
)

print(f"Final derivative peak radius for 1 nm padding: {peak_1nm:.2f} Å")
print(f"Final derivative peak radius for 2 nm padding: {peak_2nm:.2f} Å")
print(f"Final derivative peak radius for 3 nm padding: {peak_3nm:.2f} Å")

# ─── Bar chart ───────────────────────────────────────────────────────────────
labels = ["1 nm padding", "2 nm padding", "3 nm padding"]
values = [peak_1nm, peak_2nm, peak_3nm]

plt.figure(figsize=(7, 6))
plt.bar(labels, values)

plt.xlabel("Simulation Padding")
plt.ylabel("Radius of Final Peak in dg(r)/dr (Å)")
plt.title("Final RDF Derivative Peak vs Padding")

plt.tight_layout()
plt.savefig("20260415_final_derivative_peak_vs_padding.svg")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(r2, dg2, linestyle="--", label="dg(r)/dr")
plt.plot(r2[peaks2], dg2[peaks2], "o", label="Detected peaks")
plt.xlabel("Radius (Å)")
plt.ylabel("dg(r)/dr")
plt.title("Detected Peaks in RDF Derivative (2 nm)")
plt.legend()
plt.tight_layout()
plt.savefig("20260415_detected_peaks_2nm.svg")
plt.show()