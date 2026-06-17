import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Load target vector, truncate to 8 qubits, normalise
amp = np.load("amplitude_vector.npz")['vector']
n_qubits = 8
target_len = 2 ** n_qubits
target = np.zeros(target_len, dtype=np.float64)
target[:min(len(amp), target_len)] = amp[:target_len]
target /= np.linalg.norm(target)

# 2. MPS ansatz (bond dim 8, 5 reps)
ansatz = RealAmplitudes(n_qubits, entanglement='linear', reps=5, insert_barriers=False)
num_params = ansatz.num_parameters
print(f"Ansatz parameters: {num_params}")

# 3. Variational optimisation (5 random starts)
best_fid = 0.0
best_params = None
for trial in range(5):
    init = np.random.uniform(0, 2 * np.pi, num_params)
    res = minimize(
        lambda p: 1.0 - np.abs(np.dot(
            Statevector.from_instruction(ansatz.assign_parameters(p)).data.conj(),
            target
        )) ** 2,
        init,
        method='COBYLA',
        options={'maxiter': 800}
    )
    fid = 1.0 - res.fun
    print(f"Trial {trial + 1}: fidelity {fid:.4f}")
    if fid > best_fid:
        best_fid = fid
        best_params = res.x
print(f"Best approximation fidelity: {best_fid:.4f}")

# 4. Build optimised circuit
qc_opt = ansatz.assign_parameters(best_params)

# 5. Transpile for ibm_fez + measurements
service = QiskitRuntimeService(name="qdit-ibm")
backend = service.backend("ibm_fez")
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
qc_transpiled = pm.run(qc_opt)
qc_transpiled.measure_all()
print(f"Transpiled depth: {qc_transpiled.depth()}, gate count: {qc_transpiled.size()}")

# 6. Submit to real ibm_fez QPU
print("Submitting to ibm_fez QPU...")
sampler = Sampler(mode=backend)
job = sampler.run([qc_transpiled], shots=2000)
print(f"Job ID: {job.job_id()}")
result = job.result()

# 7. Extract integer counts (V2 API – BitArray → dict[int, int])
pub_result = result[0]
raw_counts = pub_result.data.meas.get_int_counts()      # keys are integers

measured_probs = np.zeros(target_len)
for idx, cnt in raw_counts.items():
    if idx < target_len:               # safety clipping
        measured_probs[idx] = cnt / 2000

# 8. Hellinger fidelity
def hellinger(p, q):
    return (1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) ** 2

ideal_probs = target ** 2
total_fid = hellinger(ideal_probs, measured_probs)
print(f"Total fidelity (8 qubits, real ibm_fez): {total_fid:.4f}")

# 9. Save
np.savez("ibm_fez_8q_real_results.npz",
         approximation_fidelity=best_fid,
         total_fidelity=total_fid,
         backend=backend.name,
         n_qubits=n_qubits)
plt.figure()
plt.bar(["Approximation", "Real QPU total"], [best_fid, total_fid], color=["blue", "orange"])
plt.ylim(0, 1.1)
plt.title(f"8‑qubit Ocean Encoding – Real {backend.name}\n"
          f"Approx = {best_fid:.3f}, Total = {total_fid:.3f}")
plt.savefig("ibm_fez_8q_real_fidelity.png")
print("Results saved.")
