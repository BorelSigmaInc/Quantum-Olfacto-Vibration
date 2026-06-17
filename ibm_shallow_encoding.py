import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import matplotlib.pyplot as plt

# 1. Load target amplitude vector
amp = np.load("amplitude_vector.npz")['vector']
N = len(amp)
n_qubits = int(np.ceil(np.log2(N)))
padded_len = 2**n_qubits
padded = np.zeros(padded_len, dtype=np.float64)
padded[:N] = amp
padded /= np.linalg.norm(padded)
ideal_probs = padded**2

# 2. Binary‑tree controlled‑RY algorithm (Möttönen)
def amplitudes_to_circuit(v):
    """ v: real unit vector of length 2^n. Returns QuantumCircuit. """
    n = int(np.log2(len(v)))
    qc = QuantumCircuit(n)
    # Recursively apply controlled rotations
    def recurse(sub_vec, qubit, control_qubits):
        """ Apply gates on `qubit` controlled by `control_qubits` to load sub_vec. """
        half = len(sub_vec) // 2
        if half == 0:
            return
        # Split into left (0) and right (1) parts
        left = sub_vec[:half]
        right = sub_vec[half:]
        # Compute rotation angle
        norm_left = np.sum(np.abs(left)**2)
        norm_right = np.sum(np.abs(right)**2)
        theta = 2 * np.arctan2(np.sqrt(norm_right), np.sqrt(norm_left))
        # Apply gate on qubit, controlled by control_qubits
        if control_qubits:
            qc.mcry(theta, control_qubits, [qubit])
        else:
            qc.ry(theta, qubit)
        # Recurse: for the left branch (qubit = 0), we need to apply rotations on subsequent qubits
        # conditioned on qubit being 0. For right branch, conditioned on qubit being 1.
        # We implement this by adding the current qubit to the control list with the appropriate control state.
        # For left branch, we need to invert the control on the current qubit (i.e., apply when qubit = 0).
        # We can achieve that by applying X before and after the multi-controlled gate.
        # Left branch (qubit = 0):
        qc.x(qubit)
        recurse(left, qubit+1, control_qubits + [qubit])
        qc.x(qubit)
        # Right branch (qubit = 1): normal control
        recurse(right, qubit+1, control_qubits + [qubit])

    recurse(v, 0, [])
    return qc

qc = amplitudes_to_circuit(padded)
print(f"Raw circuit depth: {qc.depth()}, gate count: {qc.size()}")

# 3. Transpile for ibm_marrakesh
service = QiskitRuntimeService(name="qdit-ibm")
backend = service.backend("ibm_marrakesh")
print(f"Transpiling for {backend.name}...")
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
qc_transpiled = pm.run(qc)
print(f"Transpiled depth: {qc_transpiled.depth()}, gate count: {qc_transpiled.size()}")

# 4. Noisy simulation with ibm_marrakesh noise model
print("Running noisy statevector simulation...")
noise_model = NoiseModel.from_backend(backend)
sim = AerSimulator(noise_model=noise_model, method='statevector')
qc_meas = qc_transpiled.copy()
qc_meas.measure_all()
job = sim.run(qc_meas, shots=40000)
counts = job.result().get_counts()

sim_probs = np.zeros(2**n_qubits)
for bitstr, cnt in counts.items():
    idx = int(bitstr, 2)
    if idx < len(sim_probs):
        sim_probs[idx] = cnt / 40000

# 5. Hellinger fidelity
def hellinger_fidelity(p, q):
    return (1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))**2

fid = hellinger_fidelity(ideal_probs, sim_probs)
print(f"Noisy‑simulation fidelity: {fid:.4f}")

# 6. Save results
np.savez("ibm_shallow_results.npz", fidelity=fid, backend=backend.name)
plt.figure()
plt.bar(["Ideal", "Noisy"], [1.0, fid], color=["green", "orange"])
plt.ylim(0, 1.1)
plt.title(f"Shallow Encoding – {backend.name} Noise Model\nFidelity = {fid:.3f}")
plt.savefig("ibm_shallow_fidelity.png")
print("Results saved: ibm_shallow_results.npz, ibm_shallow_fidelity.png")
