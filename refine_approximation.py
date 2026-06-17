import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
import pickle, time

# Load target vector, truncate to 8 qubits
amp = np.load("amplitude_vector.npz")['vector']
n_qubits = 8
target_len = 2**n_qubits
target = np.zeros(target_len, dtype=np.float64)
target[:min(len(amp), target_len)] = amp[:target_len]
target /= np.linalg.norm(target)

# Build ansatz with higher expressivity
ansatz = RealAmplitudes(n_qubits, entanglement='linear', reps=10, insert_barriers=False)
num_params = ansatz.num_parameters
print(f"Ansatz parameters: {num_params}")

# Variational optimisation with 10 restarts
best_fid = 0.0
best_params = None
start_time = time.time()
for trial in range(10):
    init = np.random.uniform(0, 2*np.pi, num_params)
    def cost(params):
        qc = ansatz.assign_parameters(params)
        sv = Statevector.from_instruction(qc)
        fid = np.abs(np.dot(sv.data.conj(), target))**2
        return 1.0 - fid
    res = minimize(cost, init, method='COBYLA', options={'maxiter':1500})
    fid = 1.0 - res.fun
    print(f"Trial {trial+1}: fidelity {fid:.4f}")
    if fid > best_fid:
        best_fid = fid
        best_params = res.x
elapsed = time.time() - start_time
print(f"Optimisation finished in {elapsed:.0f}s")
print(f"Best approximation fidelity: {best_fid:.4f}")

# Save parameters and fidelity
np.savez("best_8q_params.npz", params=best_params, fidelity=best_fid)
print("Saved best_8q_params.npz")
