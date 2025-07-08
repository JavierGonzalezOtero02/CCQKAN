import numpy as onp
from numpy.polynomial.chebyshev import Chebyshev
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylane import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc

###########################################################    
class CCQKAN:
    
    def __init__(self, network_structure, degree_expansions, GFCF=False, eta=1, alpha=1, train_gfcf=True, train_angles=True, range_values=[-1,1], range_values_output=[-1.0,1.0], parameters_initialization='random', device_name='default.qubit'):

        # ----- Nertwork hyperparameters -----
        self._network_structure = network_structure
        self._degree_expansions = degree_expansions
        self._number_layers = len(self._network_structure) - 1
        self._gfcf_comp = GFCF
        self._eta = eta
        self._train_gfcf = train_gfcf
        self._train_angles = train_angles
        self._range_values = range_values

        # ----- Wires related -----
        self._N_list = self._network_structure[:-1] # This will serve to form the sizes and wires of the weight matrices. The length of this list is the number of layers
        self._K_list = self._network_structure[1:] # Same
        self._layers_wires = self._obtain_layers_wires() # Create dictionary containing the naming of all wires
        self._wires_list = self._create_wires_list() # Wires in numerical list format (added an extra wire for Hadamard test)

        # ----- Parameters related -----
        for i in range(self._number_layers):
            setattr(self, f'_parameters_Alayer_{i}', None)
        setattr(self, f'_parameters_Breconstruction', None)
        if self._gfcf_comp:
            setattr(self, f'_parameters_Cgfcf', None)
        setattr(self, f'_parameters_Dangles', None)
        self._count_layer = [0]*self._number_layers
        self._initialize_parameters(parameters_initialization, range_values_output, alpha)

          
        # ----- Circuit related -----
        self._degree_expansionsevice = qml.device(device_name, wires=self._wires_list)
        self._circuit = self._build_circuit()
    

    ##################### WIRES FUNCTIONS #####################
    def _obtain_layers_wires(self):
        """
        Computes and assigns the quantum wires used in each layer of the network.
    
        For each layer, a dictionary is created that maps semantic labels to specific wire indices.
        The wire categories for each layer include:
            - 'ad': ancillary qubits for the LCU step (same size across layers).
            - 'aw': workspace ancillas, sized based on input and output dimensions.
            - 'cheb': a single auxiliary wire, used for QSVT step.
            - 'ax': wires used to store output from the previous layer (auxiliary state).
            - 'n' : wires holding the input to the current layer.
            - 'k' : wires that encode the output of the current layer.
    
        The wire allocation is done sequentially using `last_wire` as the global counter.
        The first layer is treated specially since it has no prior layer to inherit `ax` or `n` wires from.
    
        Returns:
            dict: A dictionary mapping each layer index to its corresponding wire assignments.
                  Each layer dictionary maps the categories 'ad', 'aw', 'cheb', 'ax', 'n', and 'k' to lists of wire indices.
        """

        # nº anzilla qubits for the lcu step (all layers will have the same number of lcu qubits)
        lcu = int(np.ceil(np.log2(self._degree_expansions + 1)))
        # Prepare auxiliary iterables
        last_wire = 0
        last_k = 0

        # Dict of qubits to return
        layers_wires = {}
        
        for layer in range(self._number_layers):
            layer_wires = {}
            
            # Compute k for this layer. k is the output size of this layer
            if int(np.ceil(np.log2(self._network_structure[layer+1]))) == 0:
                k = 1
            else:
                k = int(np.ceil(np.log2(self._network_structure[layer+1])))
            
            # Special case when layer is 0
            if layer == 0:
                # Compute n for this layer. n is the input size of this layer
                if int(np.ceil(np.log2(self._network_structure[layer]))) == 0:
                    n = 1
                else:
                    n = int(np.ceil(np.log2(self._network_structure[layer])))
                
                # Add wires to wires dictionary
                layer_wires['ad'] = list(range(last_wire, last_wire + lcu))
                last_wire = last_wire + lcu
        
                layer_wires['aw'] = list(range(last_wire, last_wire + n + k))
                last_wire = last_wire + n + k

                layer_wires['cheb'] = list(range(last_wire, last_wire + 1))
                last_wire = last_wire + 1
                
                layer_wires['ax'] = list(range(last_wire, last_wire + n))
                last_wire = last_wire + n
                
                layer_wires['n'] = list(range(last_wire, last_wire + n))
                last_wire = last_wire + n
                
                layer_wires['k'] = list(range(last_wire, last_wire + k))
                last_wire = last_wire + k
        
            # Rest of layers
            else:
                n = last_k
        
                # Add wires to wires dictionary
                layer_wires['ad'] = list(range(last_wire, last_wire + lcu))
                last_wire = last_wire + lcu
        
                layer_wires['aw'] = list(range(last_wire, last_wire + n + k))
                last_wire = last_wire + n + k

                layer_wires['cheb'] = list(range(last_wire, last_wire + 1))
                last_wire = last_wire + 1
        
                layer_wires['ax'] = layers_wires[layer - 1]['ad'] + layers_wires[layer - 1]['aw'] + layers_wires[layer - 1]['ax'] + layers_wires[layer - 1]['n']
        
                layer_wires['n'] = layers_wires[layer - 1]['k']
        
                layer_wires['k'] = list(range(last_wire, last_wire + k))
                last_wire = last_wire + k
                
                
            # Actualize last k value
            last_k = k
        
            # Add wires of this layer to wires dictionary
            layers_wires[layer] = layer_wires
        return layers_wires

    def _create_wires_list(self):
        """
        Creates a sorted list of all qubit wires used across all layers in the network.
    
        This function collects all wire indices from each layer in `self._layers_wires`,
        merges them into a unified set to avoid duplicates, and sorts the result.
        Additionally, it appends one extra wire index at the end, which is reserved
        for use in the Hadamard test (e.g., as a control or ancilla qubit).
    
        Returns:
            list: A sorted list of unique wire indices used by the circuit,
                  including one additional wire for auxiliary purposes.
        """
        wires_list = set()
        for layer_wires in self._layers_wires.values():
            for wires in layer_wires.values():
                wires_list.update(wires)
        wires_list = sorted(wires_list)
    
        # Add one extra qubit for the Hadamard test (e.g., control wire)
        wires_list.append(wires_list[-1] + 1)
    
        return wires_list

    ##################### PARAMETERS FUNCTIONS #####################
    def _initialize_parameters(self, parameters_initialization, initialization_range_reconstruction, alpha):
        """
        Initializes the variational parameters for each layer in the quantum network.
    
        Parameters:
            parameters_initialization (str): Strategy for initialization. 
                - 'half': initializes all diagonal elements to 0.5.
                - otherwise: initializes each diagonal element uniformly in [-1, 1].
            initialization_range_reconstruction (list or array): Parameters used for output rescaling, 
                typically defining the target reconstruction range.
    
        Returns:
            list: A list containing:
                - A list of parameter arrays for each layer (each of shape (d+1, N*K)).
                - A final array for output reconstruction (typically length 2).
        """  
        # Weight parameters
        for layer in range(self._number_layers):
            if parameters_initialization == 'half':
                setattr(self, f'_parameters_Alayer_{layer}', np.array([np.eye(self._N_list[layer] * self._K_list[layer]) * 0.5 for i in range(self._degree_expansions + 1)], requires_grad=True))   
                
            else:
                setattr(self, f'_parameters_Alayer_{layer}', np.array([
                    np.random.uniform(-1, 1, self._N_list[layer] * self._K_list[layer])
                    for _ in range(self._degree_expansions + 1)
                ], requires_grad=True))                

        # Reconstruction parameters
        setattr(self, f'_parameters_Breconstruction', np.array(initialization_range_reconstruction, requires_grad=True))

        # GFCF values
        if self._gfcf_comp:
            if self._train_gfcf:
                setattr(self, f'_parameters_Cgfcf', np.array([alpha], requires_grad=True))
            else:
                setattr(self, f'_parameters_Cgfcf', np.array([alpha], requires_grad=False))

        polynomials = []
        for i in range(1, self._degree_expansions + 1):
            # Convert Chebyshev basis polynomial to standard monomial basis
            coeffs = Chebyshev.basis(i).convert(kind=onp.polynomial.Polynomial).coef.tolist()
            polynomials.append(coeffs)

        angles_layers = []    
        for layer in range(self._number_layers):
            # Skip degree-0 polynomial (identity)
            angles = []
            for polynomial in polynomials:
                # Convert polynomial coefficients to QSVT phase angles
                for angle in qml.poly_to_angles(polynomial, "QSVT").tolist():
                    angles.append(angle)
            angles_layers.append(angles)

        if self._train_angles:
            angles_layers = np.array(angles_layers, requires_grad=True)
        else:
            angles_layers = np.array(angles_layers, requires_grad=False)
        setattr(self, f'_parameters_Dangles', angles_layers)
            
                


    def _set_parameters(self, *parameters):
        """
        Assigns a complete set of variational parameters to the model.
    
        This method stores the provided `parameters` list within the network, which typically includes
        the layer-wise matrices used in the QKAN circuit and potentially additional
        reconstruction or scaling parameters.
    
        Args:
            parameters (list): A collection of parameters, formatted according to the model's internal architecture.
        """
        for layer in range(self._number_layers):
            setattr(self, f'_parameters_Alayer_{layer}', parameters[layer])
        if self._gfcf_comp:
            setattr(self, f'_parameters_Breconstruction', parameters[-3])
            setattr(self, f'_parameters_Cgfcf', parameters[-2])
        else:
            setattr(self, f'_parameters_Breconstruction', parameters[-2])
        setattr(self, f'_parameters_Dangles', parameters[-1])    

    
    
    ##################### QKAN CONSTRUCTION FUNCTIONS #####################

    def _CHEB(self, matrix, index, layer):
        """
        Implements the QSVT (Quantum Singular Value Transformation) used within the CHEB step in QKAN framework.
    
        This function applies the Chebyshev polynomial transformation (the CHEB step) to the singular
        values of a block-encoded matrix. It is a core component of QKAN,
        responsible for activating or modulating the singular values of the input via a Chebyshev polynomial
        of a given degree.
    
        The method uses a sequence of unitary operations interleaved with controlled phase rotations 
        (encoded as precomputed projectors). The sequence corresponds to the QSVT procedure, alternating 
        between applying the transformation and its adjoint. For the input layer, it uses a direct block-encoding 
        of the input matrix. For deeper layers, it uses a recursive SUM step to simulate data propagation.
    
        Args:
            matrix (array-like): The matrix (block-encoded operator) to be transformed.
            index (int): Index of the Chebyshev polynomial term to apply (starting from 1).
                         Index 0 (identity) is excluded and handled elsewhere.
            layer (int): The index of the current QKAN layer where the QSVT step is being applied.
        """
        normal = True  # Flag to determine whether to apply forward or adjoint encoding
    
        # Apply each projector and the corresponding encoding step
        for _ in range(index):
            qml.PCPhase(self._parameters_Dangles[layer][self._count_layer[layer]], dim=self._N_list[layer], wires=self._layers_wires[layer]['cheb'] + self._layers_wires[layer]['ax'] + self._layers_wires[layer]['n'])
            self._count_layer[layer] += 1
            if layer == 0:
                # Input layer: apply block-encoding directly
                if normal:
                    qml.BlockEncode(matrix, wires=self._layers_wires[layer]['ax'] + self._layers_wires[layer]['n'])
                else:
                    qml.adjoint(qml.BlockEncode)(matrix, wires=self._layers_wires[layer]['ax'] + self._layers_wires[layer]['n'])
            else:
                # Hidden/output layers: apply the SUM operator instead
                if normal:
                    self._SUM(matrix, layer - 1)
                else:
                    qml.adjoint(self._SUM)(matrix, layer - 1)
    
            normal = not normal
    
        # Final projector applied after all alternating steps
        qml.PCPhase(self._parameters_Dangles[layer][self._count_layer[layer]], dim=self._N_list[layer], wires=self._layers_wires[layer]['cheb'] + self._layers_wires[layer]['ax'] + self._layers_wires[layer]['n'])
        self._count_layer[layer] += 1
    def _MUL(self, matrix, index, layer):
        """
        Applies the multiplication step for a specific Chebyshev polynomial term in the LCU decomposition.
    
        This function performs two main actions:
        1. Applies the Chebyshev polynomial transformation using the `_CHEB` method.
        2. Applies a block-encoded operation using the parameters corresponding to the given term index.
    
        Args:
            matrix (array-like): The matrix or data input passed to the Chebyshev transformation.
            index (int): The index of the polynomial term and associated parameter matrix.
            layer (int): The index of the layer in which this operation is being applied.
        """
        # Apply Chebyshev transformation for the specified term

        if index == 0:
            qml.Identity(wires=self._layers_wires[layer]['cheb'] + self._layers_wires[layer]['ax'] + self._layers_wires[layer]['n'])
        else:
            self._CHEB(matrix, index, layer)


        # Apply block-encoded unitary with the parameters for this term
        qml.BlockEncode(np.diag(getattr(self, f'_parameters_Alayer_{layer}')[index]), self._layers_wires[layer]['aw'] + self._layers_wires[layer]['n'] + self._layers_wires[layer]['k'])

    
    def _LCU(self, matrix, layer):
        """
        Implements a Linear Combination of Unitaries (LCU) step for a given layer.
    
        This function prepares an equal superposition over the 'ad' wires (degree selection qubits),
        and then conditionally applies the `_MUL` operation based on binary encodings of each
        Chebyshev polynomial term index. After applying all conditional operations, the 'ad' wires
        are reset into superposition again.
    
        Args:
            matrix (array-like): Input matrix used within the `_MUL` operation.
            layer (int): The index of the current layer in the network architecture.
        """
        # Prepare auxiliary 'ad' wires in equal superposition
        for wire in self._layers_wires[layer]['ad']:
            qml.Hadamard(wires=wire)
    
        # Iterate over all polynomial terms (including the 0th term)
        for i in range(self._degree_expansions + 1):
            # Binary representation of i to control application of specific polynomial
            bin_list = list(format(i, f'0{len(self._layers_wires[layer]["ad"])}b'))
            bin_list.reverse()
            bin_list = [int(b) for b in bin_list]
    
            # Apply controlled multiplication operation based on 'ad' controls
            qml.ctrl(self._MUL, control=self._layers_wires[layer]['ad'], control_values=bin_list)(matrix, i, layer)

    
        # Re-apply Hadamard to reset 'ad' wires
        for wire in self._layers_wires[layer]['ad']:
            qml.Hadamard(wires=wire)
        
    def _SUM(self, matrix, layer):
        """
        STEP 6 of QKANLayer. Applies a summation using Hadamard gates and an LCU.
    
        This function prepares a uniform superposition over the input wires of the
        given layer (denoted by 'n') using Hadamard gates, simulates a Linear Combination
        of Unitaries (LCU) via `self.LCUSimulator(matrix)`, and then reverses the superposition
        by applying another round of Hadamard gates.
    
        Args:
            matrix (array-like): Matrix or parameters used in the LCU simulation.
            layer (int): The index of the layer for which to apply the SUM operation.
                         The wires used are retrieved from `self._wires_layers[layer]['n']`.
    
        """
        self._count_layer[layer] = 0
        # Prepare uniform superposition on input wires
        for wire in self._layers_wires[layer]['n']:
            qml.Hadamard(wires=wire)

        # Apply LCU simulation using the provided matrix
        self._LCU(matrix, layer)

    
        # Reverse the superposition
        for wire in self._layers_wires[layer]['n']:
            qml.Hadamard(wires=wire)
        
    def _build_circuit(self):
        """        
        This method builds a quantum circuit designed to perform a Hadamard test which is the main piece of QKAN,
        for estimating the real part of the expectation value ⟨ψ|Uqkan|ψ⟩, where:
        - |ψ⟩ is a computational basis state corresponding to the binary representation
          of `dimension`, loaded on a designated subset of qubits.
        - U is a unitary operation defined by `self._Uqkan`, applied conditionally
          based on the control qubit.
    
        The circuit proceeds as follows:
        1. Encodes the binary representation of `dimension` onto the qubits specified in
           `self._layer_wires[0]['k']`, padded with zeros as needed.
        2. Applies a Hadamard gate to the control qubit (the first wire in `self._wires_list`).
        3. Applies the unitary `Uqkan` controlled by that qubit.
        4. Applies another Hadamard gate and measures the expectation value of Pauli-Z.
    
        Returns:
            function: A QNode representing the Hadamard test, accepting inputs:
                - matrix (array): Parameters or data to be passed into the unitary `Uqkan`.
                - dimension (int): The integer to encode as a computational basis state.
        """
        @qml.qnode(self._degree_expansionsevice, wires=self._wires_list, cache=False)
        def circuit(matrix, dimension):
            state = np.array([float(bit) for bit in bin(dimension)[2:]], requires_grad=False)
            if len(state) < len(self._layers_wires[self._number_layers - 1]['k']):
                padding = [0.0] * (len(self._layers_wires[self._number_layers - 1]['k']) - len(state))
                state = np.array(padding + list(state), requires_grad=False)
            qml.BasisState(state, wires=self._layers_wires[self._number_layers - 1]['k'])
            qml.Hadamard(wires=self._wires_list[-1])
            qml.ctrl(self._SUM, control=self._wires_list[-1], control_values=1)(matrix, self._number_layers - 1) # index of layer starts by 0
            qml.Hadamard(wires=self._wires_list[-1])
    
            return qml.expval(qml.PauliZ(wires=self._wires_list[-1]))
        return circuit



    ##################### FORWARD AND INPUT PREPARATION FUNCTIONS #####################
    
    def forward(self, X, *parameters):
        """
        Forward pass of QKAN.
    
        Applies the quantum circuit to each input sample in `X`, using the specified 
        circuit parameters. Each sample is transformed into a diagonal matrix, passed 
        through the quantum model, and mapped to an output vector of dimension `K`, 
        where `K = self._network_structure[-1]`.
    
        This method supports two output decoding modes:
        - GFCF-compatible decoding
        - Standard rescaling from [-1, 1] to the original value range
    
        Parameters:
            X (array-like): Input matrix of shape (num_samples, N). If only one sample is provided,
                            it must still be wrapped as a 2D array (e.g., `X = [[3]]`).
            *parameters: Parameters required by the quantum circuit. These are passed to `set_parameters`.
    
        Returns:
            list of np.ndarray: Output matrix of shape (num_samples, K). If only one sample and one
                                output dimension are present, the result is still returned as a 2D array (e.g., `Y = [[4]]`).
        """
        self._set_parameters(*parameters)  # Required for Pennylane compatibility during training
    
        Y = []  # Collect output vectors for each sample
        for sample in X:
            for i in range(len(self._count_layer)):
                self._count_layer[i] = 0
            matrix = self._prepareSample(sample)
            mapped_output = []
    
            # Compute output value for each output dimension
            for dimension in range(self._network_structure[-1]):

                result = self._circuit(matrix, dimension)
                mapped_output.append(result)               

    
            # Rescale the output to the original value range
            mapped_output = np.array(mapped_output, requires_grad=False)

    
            if self._gfcf_comp:
                # GFCF decoding: inverse map from non-linear space
                output = ((mapped_output * (self._parameters_Breconstruction[1] - self._parameters_Breconstruction[0])) / self._eta) + self._parameters_Breconstruction[0] # Parameters [range recons, eta, alpha]
                          
            else:
                # Standard decoding: from [-1, 1] to original space
                output = ((mapped_output + 1) / 2) * (self._parameters_Breconstruction[1] - self._parameters_Breconstruction[0]) + self._parameters_Breconstruction[0]            
            Y.append(output)
            gc.collect()
        return Y
        

    ## ------ Input preparation -------##
    def _prepareSample(self, array_input):
        """
        Prepares a diagonal matrix based on a normalized version of the input array that can fit into the network.
    
        This method normalizes the input array based on the user-defined range specified
        by `self._range_values`. It supports two modes:
        
        - GFCF-compatible mode (`self._gfcf_compatible` is True): Applies the non-linear transformation proposed by  Parand and
        Delkhosh (2017) GFCFs
        - Standard mode: Applies linear scaling to the interval [-1, 1] so Chebyshev Polynomials can be used.
    
        Args:
            array_input (array-like): Input data array to be converted into a diagonal matrix.
    
        Returns:
            np.ndarray: Diagonal matrix where each diagonal entry is the scaled version of 
            the corresponding element in `array_input`.
    
        Raises:
            ValueError: If any value in `array_input` is not strictly within the specified
            range (`self._range_values`), or if the GFCF parameters are inconsistent.
        """
        # Convert to NumPy array (non-differentiable)
        array_input = np.array(array_input, requires_grad=False)
    
        # Compute statistics of the input
        min_val = np.min(array_input)
        max_val = np.max(array_input)
    
        # Allowed scaling range
        minimum, maximum = self._range_values
    
        # Sanity check: ensure maximum > minimum to avoid division by zero
        if maximum <= minimum:
            raise ValueError("Range values are invalid: maximum must be strictly greater than minimum.")
    
        # Check that values are strictly within the allowed range
        if min_val <= minimum or max_val >= maximum:
            raise ValueError("Input array values must be STRICTLY inside the defined range (exclusive).")
    
        if self._gfcf_comp:
            # GFCF-specific input scaling: linear scaling to [0, η] followed by a non-linear map
            if maximum < self._eta:
                raise ValueError("The eta parameter must be less than or equal to the maximum allowed value.")
    
            normalized_input = self._eta * (array_input - minimum) / (maximum - minimum)
            gfcf_input = 1 - 2 * (normalized_input / self._eta) ** self._parameters_Cgfcf[0]
            return np.diag(gfcf_input)
    
        else:
            # Standard linear normalization to [-1, 1]
            normalized_input = 2 * (array_input - minimum) / (maximum - minimum) - 1
            return np.diag(normalized_input)    