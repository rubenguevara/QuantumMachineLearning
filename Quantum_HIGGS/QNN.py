
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
    for i, x in np.ndenumerate(data):
        circuit.rx(x, data_reg[i[1]])
    
    return circuit
    

def ansatz(circuit, data_reg, theta, n_qubits):
    """
    Simple ansatz from Wold. 
    5.2
    """
    for i in range(n_qubits-1):                 
        circuit.cx(data_reg[i], data_reg[i+1])              # Entanlging qubits using CNOT
        
    for i, w in np.ndenumerate(theta):                      # Applying R_y rotations to each qubit
        circuit.ry(w, data_reg[i[1]])
    
    return circuit

def parity(counts, shots, n_qubits):
    """
    Parity operator for binary output. 
    5.3
    """
    output = 0
    for bitstring, sample in counts.items():
        state = [int(i) for i in bitstring[0:n_qubits]]
        parity = sum(state) % 2
        if parity==1:
            output += sample
    output /= shots
    return output

def qnn(x, theta, backend = qk.Aer.get_backend('qasm_simulator'), shots=1000):
    """
    x: data input
    theta: parameters to train
    backend: for realistic noise. Can choose 'nairobi', 'manila' or leave blank ('qasm_simulator)
    shots: how many times we measure the final state
    """
    n_qubits = len(x[1])
    data_reg = qk.QuantumRegister(n_qubits)
    clas_reg = qk.ClassicalRegister(n_qubits)
    circuit = qk.QuantumCircuit(data_reg, clas_reg)
    
    circuit = qubit_encoder(circuit, data_reg, x)           # 5.1
    circuit = ansatz(circuit, data_reg, theta, n_qubits)    # 5.2
    circuit.measure_all()
    #print(circuit)
    
    if backend == 'nairobi':
        backend = backend_nairobi
    elif backend == 'manila':
        backend = backend_manila
    
    job = qk.execute(circuit,
                    backend, 
                    shots = shots)
    counts = job.result().get_counts(circuit)
    
    y_pred = parity(counts, shots, n_qubits)                # 5.3
    return y_pred, counts


def gradient(x, theta, shots):
    deriv_plus = np.zeros_like(theta)
    deriv_minus = np.zeros_like(theta)
    
    for i in tqdm(range(len(theta))):
        theta[i] += np.pi/2                                 # Parameter shifted forward
        deriv_plus[i],c = qnn(x, theta, shots=shots)
        
        theta[i] -= np.pi                                   # Parameter shifted backwards
        deriv_minus[i],c = qnn(x, theta, shots=shots)
        
        theta[i] += np.pi/2                                 # Parametter reset
        
    return 0.5*(deriv_plus - deriv_minus)                   # Linear combination

def train(x_list, y_list, theta, lr, epochs, shots):
    loss = []
    for i in tqdm(range(epochs)):
        grad = np.zeros(len(theta))
        loss.append(0)  
        
        for y in tqdm(y_list):
            y_pred, c = qnn(x_list, theta, shots=shots)     # Prediction
            loss[-1] += (y_pred - y)**2                     # Accumulative loss
            grad = grad + (y_pred - y)*gradient(x_list, theta, shots=shots)
        
        loss[-1] = loss[-1]/len(y)                          # Normalize
        grad = grad/len(y)
        theta += -lr*grad                                   # Update parameters
    
    return theta, loss


