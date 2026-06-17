#!/usr/bin/env python3
"""OCNQENC – Generate individual panel PNGs A–J."""
import numpy as np, xarray as xr, os, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Load data ----
data_dir = "/home/rrmr/Desktop/RRMR/BProjects/BorelSIgmaInc/Projects/Research Projects/Kryptur/OCNQENC/sst_data"
files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
daily_fields = []
for f in files:
    ds = xr.open_dataset(f, chunks='auto')
    daily_fields.append(ds['sst'].squeeze(drop=True).values.astype(np.float32))
    ds.close()
sst_cube = np.stack(daily_fields, axis=0)          # (31, 720, 1440)

# ---- Pre‑process ----
sst_anom = sst_cube - np.nanmean(sst_cube, axis=0, keepdims=True)
sst_anom = np.nan_to_num(sst_anom, nan=0.0)
T, H, W = sst_anom.shape
X = sst_anom.reshape(T, -1)

# ---- SVD ----
U, s, Vt = np.linalg.svd(X, full_matrices=False)
k = 20
subsample = 1266
spatial_modes = Vt[:k, ::subsample]
coeff_vector = spatial_modes.ravel()
amplitude_vector = coeff_vector / np.linalg.norm(coeff_vector)

# ---- Panel A ----
fig, ax = plt.subplots(figsize=(8,6))
ax.semilogy(s, 'o-', markersize=4, color='navy')
ax.set_title('A) Singular Values (log)')
ax.set_xlabel('Mode index'); ax.set_ylabel('Singular value')
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig('panel_A.png', dpi=200); plt.close(fig)

# ---- Panel B ----
fig, ax = plt.subplots(figsize=(8,6))
var_exp = s**2 / np.sum(s**2)
cum_var = np.cumsum(var_exp)
ax.plot(cum_var * 100, '.-', color='darkgreen')
ax.set_title('B) Cumulative Explained Variance')
ax.set_xlabel('Number of modes'); ax.set_ylabel('Cumulative variance (%)')
ax.axhline(y=95, color='r', linestyle='--', alpha=0.6)
ax.axvline(x=k, color='gray', linestyle='--', alpha=0.6)
ax.text(k+1, 70, f'k={k} modes', fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig('panel_B.png', dpi=200); plt.close(fig)

# ---- Panel C (4 maps) ----
fig, axes = plt.subplots(2,2, figsize=(16,10))
for i, ax in enumerate(axes.flat):
    mode_map = Vt[i, :].reshape(H, W)
    im = ax.imshow(mode_map, cmap='RdBu_r', aspect='auto', origin='lower')
    ax.set_title(f'Mode {i+1}')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
fig.suptitle('C) Top Spatial Modes')
fig.tight_layout(); fig.savefig('panel_C.png', dpi=200); plt.close(fig)

# ---- Panel D ----
fig, ax = plt.subplots(figsize=(10,5))
for i in range(5):
    ax.plot(U[:, i] * s[i], label=f'Mode {i+1}', marker='o')
ax.set_title('D) Temporal Coefficients (first 5 modes)')
ax.set_xlabel('Day (Jan 2023)'); ax.set_ylabel('Amplitude')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig('panel_D.png', dpi=200); plt.close(fig)

# ---- Panel E ----
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(amplitude_vector**2, bins=50, color='purple', edgecolor='white', alpha=0.8)
ax.set_title('E) Amplitude Probabilities Histogram')
ax.set_xlabel('Probability'); ax.set_ylabel('Count')
fig.tight_layout(); fig.savefig('panel_E.png', dpi=200); plt.close(fig)

# ---- Panel F ----
fig, ax = plt.subplots(figsize=(8,6))
sorted_probs = -np.sort(-(amplitude_vector**2))
cum_probs = np.cumsum(sorted_probs)
ax.plot(cum_probs, color='darkred')
ax.set_title('F) Cumulative Probability Mass')
ax.set_xlabel('Sorted component index'); ax.set_ylabel('Cumulative probability')
ax.axhline(y=0.95, color='gray', linestyle='--')
ax.set_xscale('log'); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig('panel_F.png', dpi=200); plt.close(fig)

# ---- Panel G ----
fig, ax = plt.subplots(figsize=(8,6))
first_day_pattern = spatial_modes[:, 0]
ax.stem(first_day_pattern)
ax.set_title('G) Encoded Coefficients (1st spatial pt)')
ax.set_xlabel('Mode'); ax.set_ylabel('Value')
fig.tight_layout(); fig.savefig('panel_G.png', dpi=200); plt.close(fig)

# ---- Panel H ----
fig, ax = plt.subplots(figsize=(8,6))
probs = amplitude_vector**2
entropy = -np.sum(probs * np.log2(probs + 1e-15))
max_entropy = np.log2(len(probs))
ax.bar(['Actual', 'Max possible'], [entropy, max_entropy], color=['steelblue', 'lightgray'])
ax.set_title(f'H) Entropy = {entropy:.2f} bits (max {max_entropy:.1f})')
ax.set_ylabel('Bits')
fig.tight_layout(); fig.savefig('panel_H.png', dpi=200); plt.close(fig)

# ---- Panel I ----
from qiskit import qpy
with open("encoding_circuit.qpy", "rb") as f:
    qc = qpy.load(f)[0]
ops = qc.decompose(reps=2).count_ops()
fig, ax = plt.subplots(figsize=(8,6))
gates = list(ops.keys()); counts = list(ops.values())
ax.barh(gates, counts, color='darkorange')
ax.set_title('I) Circuit Gate Counts')
ax.set_xlabel('Count')
fig.tight_layout(); fig.savefig('panel_I.png', dpi=200); plt.close(fig)

# ---- Panel J ----
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.sort(amplitude_vector), np.linspace(0,1,len(amplitude_vector)), color='magenta')
ax.set_title('J) Amplitude Distribution (CDF)')
ax.set_xlabel('Amplitude value'); ax.set_ylabel('Cumulative fraction')
fig.tight_layout(); fig.savefig('panel_J.png', dpi=200); plt.close(fig)

print("All panels A–J saved as individual PNGs.")
