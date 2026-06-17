import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_experiments.library import CompleteMeasFitter
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

# 4. Get raw counts
pub_result = result[0]
raw_counts = pub_result.data.meas.get_int_counts()

# 5. Readout mitigation (via qiskit-experiments)
print("Applying readout error mitigation...")
meas_fitter = CompleteMeasFitter(backend.properties(), qubit_list=range(n_qubits))

# Convert integer-keyed counts to a list of counts for each basis state
counts_list = [raw_counts.get(i, 0) for i in range(target_len)]
mitigated_counts = meas_fitter.filter.apply(counts_list)

# Build mitigated probability array
measured_probs = np.array(mitigated_counts, dtype=np.float64)
measured_probs /= measured_probs.sum()

# 6. Hellinger fidelity
def hellinger(p, q):
    return (1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))**2

ideal_probs = target**2
total_fid = hellinger(ideal_probs, measured_probs)
print(f"Approximation fidelity: {approx_fid:.4f}")
print(f"Real‑hardware fidelity (with readout mitigation): {total_fid:.4f}")

# 7. Save
np.savez("ibm_fez_mitigated_results.npz",
         approximation_fidelity=approx_fid,
         total_fidelity=total_fid,
         backend=backend.name,
         n_qubits=n_qubits)
plt.figure()
plt.bar(["Approximation", "Real QPU mitigated"], [approx_fid, total_fid],
        color=["blue", "green"])
plt.ylim(0, 1.1)
plt.title(f"8‑qubit Ocean Encoding – Real {backend.name} with mitigation\n"
          f"Approx = {approx_fid:.3f}, Total = {total_fid:.3f}")
plt.savefig("ibm_fez_mitigated_fidelity.png")
print("Results saved.")
