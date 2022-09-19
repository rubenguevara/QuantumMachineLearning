
import numpy as np
import qiskit as qk 
import pickle
from qiskit.providers.aer import QasmSimulator
from qiskit import IBMQ
from tqdm import tqdm

IBMQ.enable_account("a8c5d5c6e915e74af7006aaa6e1c55922a8995336580e6f64df2a46940b7d1aff5aa540791e20c6c9a21ad0f46d53b03b858b67252a1e3d36570d2f49ee3092d") 
provider = IBMQ.get_provider(hub="ibm-q")
backend_manila = provider.get_backend('ibmq_manila')
backend_manila = QasmSimulator.from_backend(backend_manila)
pickle.dump(backend_manila, open("backend_manila", "wb"))

backend_nairobi = provider.get_backend('ibm_nairobi')
backend_nairobi = QasmSimulator.from_backend(backend_nairobi)
pickle.dump(backend_nairobi, open("backend_nairobi", "wb"))

def qubit_encoder(circuit, data_reg, data):
    """
    R_X Qubit Encoding. 
    5.1
    """
    for (i, x) in enumerate(data):
        circuit.rx(x, data_reg[i])
    
    return circuit
    

def ansatz(circuit, data_reg, theta, n_qubits):
    """
    Simple ansatz from Wold. 
    5.2
    """
    for i in range(n_qubits-1):                 
        circuit.cx(data_reg[i], data_reg[i+1])                            # Entanlging qubits using CNOT
        
    for (i, w) in enumerate(theta):                                       # Applying R_y rotations to each qubit
        circuit.ry(w, data_reg[i])
    
    return circuit

def parity(counts, shots):
    """
    Parity operator for binary output. 
    5.3
    """
    output = 0
    for bitstring, sample in counts.items():
        state = [int(i) for i in bitstring]
        parity = sum(state) % 2
        if parity==0:
            output += sample
    output /= shots
    
    return output

def qnn(x, theta, backend, shots=1000, visual=False):
    """
    x: data input
    theta: parameters to train
    backend: for realistic noise. Can choose 'Nairobi' or 'Manila'
    shots: how many times we measure the final state
    """
    n_qubits = len(x)
    data_reg = qk.QuantumRegister(n_qubits)
    circuit = qk.QuantumCircuit(data_reg)
    
    circuit = qubit_encoder(circuit, data_reg, x)                            # 5.1
    circuit = ansatz(circuit, data_reg, theta, n_qubits)                     # 5.2
    circuit.measure_all()
    
    if visual == True:
        print(circuit)
    
    if backend == 'Nairobi':
        backend = backend_nairobi
    elif backend == 'Manila':
        backend = backend_manila
    
    job = qk.execute(circuit,
                    backend, 
                    shots = shots,
                    seed_simulator=42,
                    seed_transpiler=42)
    counts = job.result().get_counts(circuit)
    
    y_pred = parity(counts, shots)                                           # 5.3
    return y_pred, counts


def gradient(x, theta, backend, shots):
    deriv_plus = np.zeros_like(theta)
    deriv_minus = np.zeros_like(theta)
    
    for i in range(len(theta)):
        theta[i] += np.pi/2                                                  # Parameter shifted forward
        deriv_plus[i],c = qnn(x, theta, backend, shots)
        
        theta[i] -= np.pi                                                    # Parameter shifted backwards
        deriv_minus[i],c = qnn(x, theta, backend, shots)
        
        theta[i] += np.pi/2                                                  # Parametter reset
        
    return 0.5*(deriv_plus - deriv_minus)                                    # Linear combination

def train(x_list, y_list, theta, lr, epochs, shots, backend):
    """
    x_list: data input
    y_list: label of data
    theta: parameter to be trained
    lr: learing rate og GD
    epochs: how many iterations to train
    shots: how many times we measure the final state
    backend: for realistic noise. Can choose 'Nairobi', 'Manila'
    """
    
    loss = []
    for i in tqdm(range(epochs)):
        grad = np.zeros_like(theta)
        loss.append(0)  
        
        for x, y, (k, t) in zip(x_list, y_list, enumerate(theta)):
            y_pred, c = qnn(x, t, backend, shots)                            # Prediction
            loss[-1] += 0.5*(y_pred - y)**2                                  # Eq. 10
            grad[k] = grad[k] + (y_pred - y)*gradient(x, t, backend, shots)  # Eq. 13
            
            
        theta += -lr*grad                                                    # Update parameters
    
    return theta, loss


