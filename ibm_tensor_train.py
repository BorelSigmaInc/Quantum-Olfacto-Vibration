import numpy as np, pennylane as qml
from pennylane import numpy as pnp
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import matplotlib.pyplot as plt

# 1. Load amplitude vector
amp = np.load("amplitude_vector.npz")['vector']
N = len(amp)
n_qubits = int(np.ceil(np.log2(N)))
padded_len = 2**n_qubits
padded = np.zeros(padded_len, dtype=np.float64)
padded[:N] = amp
padded /= np.linalg.norm(padded)
ideal_probs = padded**2

# 2. Build MPS (tensor‑train) circuit in PennyLane
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="autograd")
def mps_circuit():
    qml.MPS(wires=range(n_qubits), n_block_wires=1, block=qml.MPSBlock, reset=True, init_state=padded)
    return qml.state()

print(f"Building MPS circuit for {n_qubits} qubits...")
state = mps_circuit()
print(f"MPS fidelity (exact): {np.abs(np.dot(state.conj(), padded))**2:.6f}")

# Convert PennyLane tape to Qiskit circuit
tape = mps_circuit.tape
qc_qiskit = QuantumCircuit(n_qubits)
# Map PennyLane operations to Qiskit
for op in tape.operations:
    if op.name == "MPS":
        continue  # MPS is a template, the tape already contains decomposed gates
    elif op.name == "CNOT":
        qc_qiskit.cx(op.wires[0], op.wires[1])
    elif op.name == "RY":
        qc_qiskit.ry(op.parameters[0], op.wires[0])
    elif op.name == "RZ":
        qc_qiskit.rz(op.parameters[0], op.wires[0])
    elif op.name == "RX":
        qc_qiskit.rx(op.parameters[0], op.wires[0])
    elif op.name == "H":
        qc_qiskit.h(op.wires[0])

print(f"Qiskit MPS circuit depth (raw): {qc_qiskit.depth()}, gates: {qc_qiskit.size()}")

# 3. Transpile for ibm_marrakesh
service = QiskitRuntimeService(name="qdit-ibm")
backend = service.backend("ibm_marrakesh")
print(f"Transpiling for {backend.name}...")
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
qc_transpiled = pm.run(qc_qiskit)
print(f"Transpiled depth: {qc_transpiled.depth()}, gates: {qc_transpiled.size()}")

# 4. Noisy simulation with statevector (much lighter than density matrix)
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

def hellinger(p, q):
    return (1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))**2

fid = hellinger(ideal_probs, sim_probs)
print(f"Noisy‑simulation fidelity: {fid:.4f}")

# 5. Save
np.savez("ibm_tt_results.npz", fidelity=fid, backend=backend.name)
plt.figure()
plt.bar(["Ideal", "Noisy MPS"], [1.0, fid], color=["green", "orange"])
plt.ylim(0,1.1)
plt.title(f"Tensor‑Train Encoding – {backend.name} Noise Model\nFidelity = {fid:.3f}")
plt.savefig("ibm_tt_fidelity.png")
print("Results saved.")
