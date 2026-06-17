#!/usr/bin/env python3
"""
OCNQENC – Step 1 (v5): 2D SVD, top‑k modes, amplitude encoding – direct statevector.
"""
import numpy as np
import xarray as xr
import requests
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# ---- 1. Download one daily OISST file (Jan 1, 2023) ----
url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/202301/oisst-avhrr-v02r01.20230101.nc"
local_file = "sst_sample.nc"
print("Downloading sample SST data...")
r = requests.get(url, stream=True)
with open(local_file, 'wb') as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

# ---- 2. Load and pre‑process ----
ds = xr.open_dataset(local_file, chunks='auto')
sst = ds['sst'].squeeze(drop=True)          # (lat=720, lon=1440)
print(f"Data shape: {sst.shape}")
sst_anomaly = sst - sst.mean(skipna=True)
arr = np.nan_to_num(sst_anomaly.values.astype(np.float32), nan=0.0)

# ---- 3. 2D SVD ----
U, s, Vt = np.linalg.svd(arr, full_matrices=False)
k = min(10, len(s))
Vt_k = Vt[:k, :]                            # (k, 1440)
coeff_vector = (np.diag(s[:k]) @ Vt_k).ravel()
print(f"Coefficient vector length: {len(coeff_vector)}  (k={k} modes × 1440 lon)")

# ---- 4. Normalise ----
norm = np.linalg.norm(coeff_vector)
amplitude_vector = coeff_vector / norm
print(f"Normalised vector norm: {np.linalg.norm(amplitude_vector):.6f}")

# ---- 5. Quantum amplitude encoding ----
N = len(amplitude_vector)
n_qubits = int(np.ceil(np.log2(N)))
padded_len = 2**n_qubits
padded_vec = np.zeros(padded_len, dtype=np.float64)
padded_vec[:N] = amplitude_vector
padded_vec /= np.linalg.norm(padded_vec)

qc = QuantumCircuit(n_qubits)
qc.initialize(padded_vec, range(n_qubits))
print(f"Circuit built with {n_qubits} qubits, depth (decomposed): {qc.decompose(reps=2).depth()}")

# ---- 6. Direct statevector computation (no simulation) ----
sv = Statevector.from_instruction(qc)
fidelity = np.abs(np.dot(sv.data.conj(), padded_vec))**2
print(f"Fidelity (direct statevector): {fidelity:.6f}")

# ---- 7. Visualisation ----
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.semilogy(s, 'o-', markersize=3)
plt.title('Singular values (2D SST)')
plt.subplot(1,2,2)
plt.bar(range(len(amplitude_vector[:50])), amplitude_vector[:50]**2)
plt.title('First 50 amplitude probabilities')
plt.tight_layout()
plt.savefig('sst_encoding_preview.png', dpi=150)
print("Saved sst_encoding_preview.png")

# ---- 8. Save circuit ----
from qiskit import qpy
with open('encoding_circuit.qpy', 'wb') as f:
    qpy.dump(qc, f)
print("Circuit saved to encoding_circuit.qpy")
