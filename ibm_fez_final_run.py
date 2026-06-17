import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt

# 1. Load target and best parameters
amp = np.load("amplitude_vector.npz")['vector']
n_qubits = 8
target_len = 2**n_qubits
target = np.zeros(target_len, dtype=np.float64)
target[:min(len(amp), target_len)] = amp[:target_len]
target /= np.linalg.norm(target)

params_data = np.load("best_8q_params.npz")
best_params = params_data['params']
approx_fid = float(params_data['fidelity'])
print(f"Approximation fidelity: {approx_fid:.4f}")

# 2. Build and transpile the circuit
ansatz = RealAmplitudes(n_qubits, entanglement='linear', reps=10, insert_barriers=False)
qc_opt = ansatz.assign_parameters(best_params)

service = QiskitRuntimeService(name="qdit-ibm")
backend = service.backend("ibm_fez")
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
qc_transpiled = pm.run(qc_opt)
qc_transpiled.measure_all()
print(f"Transpiled depth: {qc_transpiled.depth()}, gate count: {qc_transpiled.size()}")

# 3. Submit to ibm_fez (4000 shots)
print("Submitting to ibm_fez QPU (4000 shots)...")
sampler = Sampler(mode=backend)
job = sampler.run([qc_transpiled], shots=4000)
print(f"Job ID: {job.job_id()}")
result = job.result()

# 4. Get integer counts
pub_result = result[0]
raw_counts = pub_result.data.meas.get_int_counts()

measured_probs = np.zeros(target_len)
total_shots = sum(raw_counts.values())
for idx, cnt in raw_counts.items():
    if idx < target_len:
        measured_probs[idx] = cnt / total_shots

# 5. Hellinger fidelity
def hellinger(p, q):
    return (1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))**2

ideal_probs = target**2
total_fid = hellinger(ideal_probs, measured_probs)
print(f"Approximation fidelity: {approx_fid:.4f}")
print(f"Real‑hardware fidelity (raw, 4000 shots): {total_fid:.4f}")

# 6. Save
np.savez("ibm_fez_final_results.npz",
         approximation_fidelity=approx_fid,
         total_fidelity=total_fid,
         backend=backend.name,
         n_qubits=n_qubits)
plt.figure()
plt.bar(["Approximation", "Real QPU (raw)"], [approx_fid, total_fid],
        color=["blue", "orange"])
plt.ylim(0, 1.1)
plt.title(f"8‑qubit Ocean Encoding – Real {backend.name}\n"
          f"Approx = {approx_fid:.3f}, Total = {total_fid:.3f}")
plt.savefig("ibm_fez_final_fidelity.png")
print("Results saved.")
