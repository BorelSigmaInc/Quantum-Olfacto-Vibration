import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Load target vector
amp = np.load("amplitude_vector.npz")['vector']
n_qubits = int(np.ceil(np.log2(len(amp))))
full_len = 2**n_qubits
target = np.zeros(full_len, dtype=np.float64)
target[:len(amp)] = amp
target /= np.linalg.norm(target)

# 2. Build MPS ansatz with reps=3 (bond dimension 8)
qc_ansatz = RealAmplitudes(n_qubits, entanglement='linear', reps=3, insert_barriers=False)
num_params = qc_ansatz.num_parameters
print(f"Ansatz parameters: {num_params}")

# 3. Optimise to maximise fidelity
def loss(params):
    qc = qc_ansatz.assign_parameters(params)
    sv = Statevector.from_instruction(qc)
    fid = np.abs(np.dot(sv.data.conj(), target))**2
    return 1.0 - fid

print("Optimising for maximum fidelity (may take a few minutes)...")
initial_params = np.random.uniform(0, 2*np.pi, num_params)
res = minimize(loss, initial_params, method='L-BFGS-B', options={'maxiter':300})
opt_params = res.x
approx_fid = 1.0 - res.fun
print(f"Approximation fidelity (classical, bond-dimension-8 MPS): {approx_fid:.4f}")

# 4. Build optimised circuit
qc_opt = qc_ansatz.assign_parameters(opt_params)
print(f"Optimised circuit depth (raw): {qc_opt.depth()}, gates: {qc_opt.size()}")

# 5. Transpile for ibm_fez
service = QiskitRuntimeService(name="qdit-ibm")
backend = service.backend("ibm_fez")
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
qc_transpiled = pm.run(qc_opt)
print(f"Transpiled depth: {qc_transpiled.depth()}, gate count: {qc_transpiled.size()}")

# 6. Submit to real ibm_fez QPU (2000 shots, fits in free tier)
print("Submitting to ibm_fez QPU...")
sampler = Sampler(backend=backend, options=Options(shots=2000))
job = sampler.run([qc_transpiled])
print(f"Job ID: {job.job_id()}")
result = job.result()
quasi = result.quasi_dists[0]

# 7. Measured probabilities
measured_probs = np.zeros(full_len)
for idx, prob in quasi.items():
    measured_probs[int(idx)] = prob

# 8. Fidelity to the MPS approximation (ideal circuit)
ideal_state = Statevector.from_instruction(qc_opt)
ideal_probs = np.abs(ideal_state.data)**2

def hellinger(p, q):
    return (1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))**2

hardware_fid = hellinger(ideal_probs, measured_probs)
total_fid = hellinger(target**2, measured_probs)
print(f"Hardware fidelity (vs. MPS ideal): {hardware_fid:.4f}")
print(f"Total fidelity (vs. original ocean vector): {total_fid:.4f}")

# 9. Save
np.savez("ibm_fez_real_results.npz",
         approximation_fidelity=approx_fid,
         hardware_fidelity=hardware_fid,
         total_fidelity=total_fid,
         backend=backend.name)
plt.figure()
plt.bar(["MPS approx","Hardware vs MPS","Total vs ocean"],
        [approx_fid, hardware_fid, total_fid], color=["blue","cyan","orange"])
plt.ylim(0,1.1)
plt.title(f"ibm_fez Real QPU – Fidelity Breakdown")
plt.savefig("ibm_fez_real_fidelity.png")
print("Results saved.")
