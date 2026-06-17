#!/usr/bin/env python3
"""
OCNQENC – Deep visualization: SST SVD, quantum encoding diagnostics.
"""
import numpy as np
import xarray as xr
import os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- 1. Load all daily SST fields ----
data_dir = "/home/rrmr/Desktop/RRMR/BProjects/BorelSIgmaInc/Projects/Research Projects/Kryptur/OCNQENC/sst_data"
files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
daily_fields = []
for f in files:
    ds = xr.open_dataset(f, chunks='auto')
    sst = ds['sst'].squeeze(drop=True).values.astype(np.float32)
    daily_fields.append(sst)
    ds.close()
sst_cube = np.stack(daily_fields, axis=0)          # (31, 720, 1440)
print(f"SST cube shape: {sst_cube.shape}")

# ---- 2. Pre‑process ----
sst_anom = sst_cube - np.nanmean(sst_cube, axis=0, keepdims=True)
sst_anom = np.nan_to_num(sst_anom, nan=0.0)
T, H, W = sst_anom.shape
X = sst_anom.reshape(T, -1)                        # (31, 1036800)

# ---- 3. SVD ----
U, s, Vt = np.linalg.svd(X, full_matrices=False)
k = 20
print(f"Singular values computed, top 10: {s[:10]}")

# ---- 4. Encoded vector (same as pipeline) ----
subsample = 1266
spatial_modes = Vt[:k, ::subsample]                # (20, 819)
coeff_vector = spatial_modes.ravel()
amplitude_vector = coeff_vector / np.linalg.norm(coeff_vector)

# ---- 5. Prepare figure ----
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

# Panel A: Singular values (log)
ax0 = fig.add_subplot(gs[0, 0])
ax0.semilogy(s, 'o-', markersize=4, color='navy')
ax0.set_title('A) Singular Values (log)')
ax0.set_xlabel('Mode index')
ax0.set_ylabel('Singular value')
ax0.grid(True, alpha=0.3)

# Panel B: Cumulative explained variance
ax1 = fig.add_subplot(gs[0, 1])
var_exp = s**2 / np.sum(s**2)
cum_var = np.cumsum(var_exp)
ax1.plot(cum_var * 100, '.-', color='darkgreen')
ax1.set_title('B) Cumulative Explained Variance')
ax1.set_xlabel('Number of modes')
ax1.set_ylabel('Cumulative variance (%)')
ax1.axhline(y=95, color='r', linestyle='--', alpha=0.6)
ax1.axvline(x=k, color='gray', linestyle='--', alpha=0.6)
ax1.text(k+1, 70, f'k={k} modes', fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel C: Top spatial modes (maps)
for i in range(4):
    ax = fig.add_subplot(4, 3, 3 + i)  # second row, cols 0..3
    mode_map = Vt[i, :].reshape(H, W)
    im = ax.imshow(mode_map, cmap='RdBu_r', aspect='auto', origin='lower')
    ax.set_title(f'Mode {i+1}')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

# Panel D: Temporal coefficients (time series)
ax_ts = fig.add_subplot(gs[1, :])   # third row, all columns
for i in range(5):
    ax_ts.plot(U[:, i] * s[i], label=f'Mode {i+1}', marker='o')
ax_ts.set_title('D) Temporal Coefficients (first 5 modes)')
ax_ts.set_xlabel('Day (Jan 2023)')
ax_ts.set_ylabel('Amplitude')
ax_ts.legend(loc='upper right', fontsize=8)
ax_ts.grid(True, alpha=0.3)

# Panel E: Amplitude probability histogram
ax_prob = fig.add_subplot(gs[2, 0])
ax_prob.hist(amplitude_vector**2, bins=50, color='purple', edgecolor='white', alpha=0.8)
ax_prob.set_title('E) Amplitude Probabilities Histogram')
ax_prob.set_xlabel('Probability')
ax_prob.set_ylabel('Count')

# Panel F: Cumulative probability mass
ax_cum = fig.add_subplot(gs[2, 1])
sorted_probs = -np.sort(-(amplitude_vector**2))
cum_probs = np.cumsum(sorted_probs)
ax_cum.plot(cum_probs, color='darkred')
ax_cum.set_title('F) Cumulative Probability Mass')
ax_cum.set_xlabel('Sorted component index')
ax_cum.set_ylabel('Cumulative probability')
ax_cum.axhline(y=0.95, color='gray', linestyle='--')
ax_cum.set_xscale('log')
ax_cum.grid(True, alpha=0.3)

# Panel G: Encoded vector spatial pattern (first spatial point set)
ax_sp = fig.add_subplot(gs[2, 2])
# Reconstruct one spatial snapshot from the encoded modes (subsampled)
# Use the first day's spatial pattern from the encoded vector:
first_day_pattern = spatial_modes[:, 0]  # 20 values for the first subsampled point
ax_sp.stem(first_day_pattern)
ax_sp.set_title('G) Encoded Coefficients (1st spatial pt)')
ax_sp.set_xlabel('Mode')
ax_sp.set_ylabel('Value')

# Panel H: Information per qubit (entropy)
ax_ent = fig.add_subplot(gs[3, 0])
probs = amplitude_vector**2
entropy = -np.sum(probs * np.log2(probs + 1e-15))
max_entropy = np.log2(len(probs))
ax_ent.bar(['Actual', 'Max possible'], [entropy, max_entropy], color=['steelblue', 'lightgray'])
ax_ent.set_title(f'H) Entropy = {entropy:.2f} bits (max {max_entropy:.1f})')
ax_ent.set_ylabel('Bits')

# Panel I: Circuit gate distribution
from qiskit import qpy
with open("/home/rrmr/Desktop/RRMR/BProjects/BorelSIgmaInc/Projects/Research Projects/Kryptur/OCNQENC/encoding_circuit.qpy", "rb") as f:
    qc = qpy.load(f)[0]
ops = qc.decompose(reps=2).count_ops()
ax_gate = fig.add_subplot(gs[3, 1])
gates = list(ops.keys())
counts = list(ops.values())
ax_gate.barh(gates, counts, color='darkorange')
ax_gate.set_title('I) Circuit Gate Counts')
ax_gate.set_xlabel('Count')

# Panel J: Amplitude distribution (density)
ax_den = fig.add_subplot(gs[3, 2])
ax_den.plot(np.sort(amplitude_vector), np.linspace(0,1,len(amplitude_vector)), color='magenta')
ax_den.set_title('J) Amplitude Distribution (CDF)')
ax_den.set_xlabel('Amplitude value')
ax_den.set_ylabel('Cumulative fraction')

plt.suptitle('OCNQENC – Quantum Encoding of Global SST (January 2023, 14 qubits)', fontsize=16, y=0.995)
plt.savefig("/home/rrmr/Desktop/RRMR/BProjects/BorelSIgmaInc/Projects/Research Projects/Kryptur/OCNQENC/deep_report.png", dpi=200, bbox_inches='tight')
print("Comprehensive report saved as deep_report.png")
