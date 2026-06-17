import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from mitiq import zne
from mitiq.zne.inference import LinearFactory
import matplotlib.pyplot as plt

# 1. Load circuit and ideal vector
with open("encoding_circuit.qpy", "rb") as f:
    qc = qpy.load(f)[0]
amp = np.load("amplitude_vector.npz")
ideal_vector = amp['vector']
ideal_probs = np.abs(ideal_vector) ** 2

# 2. Connect and choose backend
service = QiskitRuntimeService(name="qdit-ibm")
backend = service.backend("ibm_marrakesh")
print(f"Using backend: {backend.name}")

# 3. Transpile
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
qc_transpiled = pm.run(qc)
print(f"Transpiled depth: {qc_transpiled.depth()}, gate count: {qc_transpiled.size()}")

# 4. Quantum executor for Mitiq (runs one circuit and returns probabilities)
def execute_circuit(circ, shots=40000):
    with Session(service=service, backend=backend) as session:
        sampler = Sampler(session=session, options=Options(shots=shots))
        job = sampler.run([circ])
        result = job.result()
        quasi = result.quasi_dists[0]
        probs = np.zeros(2**qc.num_qubits)
        for idx, p in quasi.items():
            probs[int(idx)] = p
        return probs

# 5. ZNE
print("Running with ZNE (scale factors 1, 2, 3)...")
zne_result = zne.execute_with_zne(
    qc_transpiled,
    execute_circuit,
    factory=LinearFactory(scale_factors=[1, 2, 3]),
    num_to_average=1,
)

# 6. Fidelity
def hellinger_fidelity(p, q):
    return (1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) ** 2

fid = hellinger_fidelity(ideal_probs, zne_result)
print(f"ZNE-mitigated fidelity: {fid:.4f}")

# 7. Save
np.savez("ibm_marrakesh_results.npz", fidelity=fid, backend=backend.name, mitigated_probs=zne_result)
plt.figure()
plt.bar(["Ideal", "ZNE-mitigated"], [1.0, fid], color=["green", "orange"])
plt.ylim(0, 1.1)
plt.title(f"IBM {backend.name} – Fidelity {fid:.3f}")
plt.savefig("ibm_marrakesh_fidelity.png")
print("Results saved.")
