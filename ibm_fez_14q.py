import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import matplotlib.pyplot as plt

# 1. Load circuit and ideal amplitude vector
with open("encoding_circuit.qpy", "rb") as f:
    qc = qpy.load(f)[0]
amp_vec = np.load("amplitude_vector.npz")['vector']
n_qubits = qc.num_qubits
full_len = 2**n_qubits

# Pad ideal vector to full 2^n
padded_ideal = np.zeros(full_len, dtype=np.float64)
padded_ideal[:len(amp_vec)] = amp_vec
padded_ideal /= np.linalg.norm(padded_ideal)
ideal_probs = padded_ideal**2

# 2. Connect and fetch ibm_fez noise model
service = QiskitRuntimeService(name="qdit-ibm")
backend = service.backend("ibm_fez")
print(f"Fetching noise model from {backend.name}...")
noise_model = NoiseModel.from_backend(backend)
print("Noise model acquired.")

# 3. Simulate with noise
print("Running noisy simulation (ideal circuit + ibm_fez noise)...")
sim = AerSimulator(noise_model=noise_model, method='statevector')
qc_meas = qc.copy()
qc_meas.measure_all()
job = sim.run(qc_meas, shots=40000)
counts = job.result().get_counts()

sim_probs = np.zeros(full_len)
for bitstr, cnt in counts.items():
    idx = int(bitstr, 2)
    if idx < full_len:
        sim_probs[idx] = cnt / 40000

# 4. Hellinger fidelity (now both arrays length 16384)
def hellinger_fidelity(p, q):
    return (1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))**2

fid = hellinger_fidelity(ideal_probs, sim_probs)
print(f"Noisy‑simulation fidelity (14 qubits, ibm_fez noise): {fid:.4f}")

# 5. Save
np.savez("ibm_fez_14q_results.npz", fidelity=fid, backend=backend.name, n_qubits=n_qubits)
plt.figure()
plt.bar(["Ideal", "Noisy (ibm_fez)"], [1.0, fid], color=["green", "orange"])
plt.ylim(0, 1.1)
plt.title(f"14‑qubit Ocean Encoding – {backend.name} Noise Model\nFidelity = {fid:.3f}")
plt.savefig("ibm_fez_14q_fidelity.png")
print("Results saved.")
