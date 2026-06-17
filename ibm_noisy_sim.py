import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import matplotlib.pyplot as plt

# 1. Load circuit and ideal vector
with open("encoding_circuit.qpy", "rb") as f:
    qc = qpy.load(f)[0]
amp = np.load("amplitude_vector.npz")
ideal_vector = amp['vector']
ideal_probs = np.abs(ideal_vector)**2

# 2. Connect to get backend noise model
service = QiskitRuntimeService(name="qdit-ibm")
backend = service.backend("ibm_marrakesh")
print(f"Fetching noise model from {backend.name}...")
noise_model = NoiseModel.from_backend(backend)
print("Noise model acquired.")

# 3. Transpile for that backend
print("Transpiling...")
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
qc_transpiled = pm.run(qc)
print(f"Transpiled depth: {qc_transpiled.depth()}, gates: {qc_transpiled.size()}")

# 4. Simulate with noise model
print("Running noisy density‑matrix simulation (this may take a few minutes)...")
sim = AerSimulator(noise_model=noise_model, method='density_matrix')
qc_meas = qc_transpiled.copy()
qc_meas.measure_all()
job = sim.run(qc_meas, shots=40000)
counts = job.result().get_counts()

# Convert to probability array
sim_probs = np.zeros(2**qc.num_qubits)
for bitstr, cnt in counts.items():
    idx = int(bitstr, 2)
    if idx < len(sim_probs):
        sim_probs[idx] = cnt / 40000

# 5. Fidelity
def hellinger_fidelity(p, q):
    return (1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))**2

fid = hellinger_fidelity(ideal_probs, sim_probs)
print(f"Noisy‑simulation fidelity (ibm_marrakesh noise model): {fid:.4f}")

# 6. Save
np.savez("ibm_noisy_sim_results.npz", fidelity=fid, backend=backend.name)
plt.figure()
plt.bar(["Ideal", "Noisy sim"], [1.0, fid], color=["green", "orange"])
plt.ylim(0, 1.1)
plt.title(f"Noisy Simulation – {backend.name} Noise Model\nFidelity = {fid:.3f}")
plt.savefig("ibm_noisy_sim_fidelity.png")
print("Results saved: ibm_noisy_sim_results.npz, ibm_noisy_sim_fidelity.png")
