

from .FHN_model import *
import numpy as np
from scipy.signal import hilbert
import warnings

class FHN_kuramoto:
    def __init__(self, model=None, vs=None, ws=None, ts=None, transient_length=100):
        if model is not None:
            self.ts = model.ts
            self.vs = model.vs
            self.ws = model.ws
            self.N = model.N
            self.block = model.block
            self.organ = model.organ
            self.transient_length=transient_length
        elif vs is not None and ws is not None:
            self.vs = vs
            self.ws = ws
            self.ts = ts
            self.N = vs.shape[1]
            self.transient_length=transient_length
        else:
            raise ValueError("Either model or vs and ws must be provided")

    
    def compute_phases(self, data):
        """
        Compute the phase of each element using the Hilbert transform.
    
        Parameters:
            data (jnp.ndarray): An (N, T) array where N is the number of elements and T is the time dimension.
        
        Returns:
            jnp.ndarray: An (N, T) array containing the phases.
        """
    
        analytic_signal = hilbert(data, axis=1)
   
        phases = jnp.angle(analytic_signal)
        return phases
    
    def kuramoto_order_parameter(self, phases):
        """
        Compute the Kuramoto order parameter for the given phases.
    
        Parameters:
            phases (jnp.ndarray): An (N, T) array of phases for N elements over T time points.
        
        Returns:
            tuple:
                - jnp.ndarray: A (T,) array representing the amplitude of the Kuramoto order parameter over time.
                - jnp.ndarray: A (T,) array representing the phase of the Kuramoto order parameter over time.
        """
        N = phases.shape[0]
    
        order_parameter_complex = jnp.sum(jnp.exp(1j * phases), axis=0) / N
    
        amplitude = jnp.abs(order_parameter_complex)
   
        phase = jnp.angle(order_parameter_complex)
        return amplitude, phase
    
    def kuramoto(self, Tfin=2000):
        """
        Function that calculates Kuramoto distinguishing between the two organs TODO : find a smart way to exclude transient/last timesteps and cleaner heart code 
        """
        Tfin = int(0.95 * len(self.ts))
    
        if self.organ == 'brain':
            phases=self.compute_phases(self.vs.T) 
            
            amplitude, phase = self.kuramoto_order_parameter(phases[:, self.transient_length:Tfin]) #here I somehow need to remove the transient and last timesteps
            self.R= jnp.mean(amplitude)
        if self.organ == 'heart':
            v_values= self.vs.T
            N_x=self.N

            block= self.block
            block=block.reshape(N_x, N_x)
            if N_x>8:
                block=~block[4:(N_x-4), 4:(N_x-4)]  #remove borders to avoid boundary effects
                v_values=v_values.reshape(N_x, N_x, -1)
                v_values=v_values[4:(N_x-4), 4:(N_x-4), :]
                v_values=v_values.reshape(-1, v_values.shape[2])
                phases=self.compute_phases(v_values)
                phases=phases[:,self.transient_length:Tfin]  #here to be changes to cover from equilibration to a bunch of stuff before end
                phases=phases.reshape(N_x-8,N_x-8,-1)
                # TODO: replace this loop
                R=[]
                for j in range(N_x-8):  # Iterate over the correct dimension

                    filtered_column = phases[:, j, :][block[:, j]]
    
                    r1, psi1=self.kuramoto_order_parameter(filtered_column)
                    R.append(r1)
                
                R=jnp.array(R)
 
                self.R= jnp.mean(R)
            else:
                warnings.warn("Too few nodes to address boundary conditions")   
                block=~block
                phases=self.compute_phases(v_values)
                phases=phases[:,self.transient_length:Tfin]  #here to be changes to cover from equilibration to a bunch of stuff before end
                phases=phases.reshape(N_x,N_x,-1)
                R=[]
                for j in range(N_x):  # Iterate over the correct dimension

                    filtered_column = phases[:, j, :][block[:, j]]
    
                    r1, psi1=self.kuramoto_order_parameter(filtered_column)
                    R.append(r1)
                
                R=jnp.array(R)
                self.R= jnp.mean(R)






class FHN_entropy:
    def __init__(self, model=None, vs=None, ws=None, ts=None, transient_length=100):
        if model is not None:
            self.ts = model.ts
            self.vs = model.vs
            self.ws = model.ws
            self.N = model.N
            self.block = model.block
            self.organ = model.organ
            self.transient_length=transient_length
        elif vs is not None and ws is not None:
            self.vs = vs
            self.ws = ws
            self.ts = ts
            self.N = vs.shape[1]
            self.transient_length=transient_length
        else:
            raise ValueError("Either model or vs and ws must be provided")


    def pattern_entropy(self, binary_arrays, s, size=None):
        """
        Calculate the entropy of the patterns contained in a set of binary arrays.
        Parameters:
        binary_arrays (jax.numpy.ndarray): A 2D array of binary arrays (N, T).
        s (int): The length of each binary array.
        Returns:
            tuple: A tuple containing:
                - float: The entropy of the binary arrays.
                - float: The normalized entropy of the binary arrays.
                - int: The length of each binary array.
        """
         # Calculate the number of unique patterns and their counts
        unique_patterns, counts = jnp.unique(binary_arrays, axis=1, return_counts=True, size=size)
        total_patterns = jnp.sum(counts)


        probabilities = (counts+1) / (total_patterns+2**s)
        unobserved= 2**s- len(counts)
    
        entropy_obs = -jnp.sum(probabilities * jnp.log2(probabilities))
        entropy_un=-unobserved/(total_patterns+2**s)*jnp.log2(1/(total_patterns+2**s))
        entropy=entropy_obs+entropy_un
    
    
        max_entropy = binary_arrays.shape[0]  # Maximum entropy for binary arrays of a given length
        normalized_entropy = entropy / max_entropy

        return entropy, normalized_entropy
            
    def handling_subsets(self, array, s, threshold=0.08): #TODO fix the problem with Tin, try to make unified code for b
        """
        This function binarizes the input array based on a threshold value, reshapes it, 
        and then calculates the entropy and normalized entropy using the `calculate_entropy_jax` function.
        Parameters:
           array (ndarray): The input array to be binarized and analyzed.
           s (int): The size parameter used for reshaping the array (length of one side of the box).
        Returns:
           tuple: A tuple containing:
               - entropy (float): The calculated entropy of the binarized array.
               - normalised (float): The normalized entropy of the binarized array.
        """
        if self.organ=='heart':
            binary_v = jnp.where(array > threshold, 1, 0)
            binary_v=binary_v.reshape(s**2, -1)
            binary_v=binary_v[:, self.transient_length:]
    
            entropy, normalised=self.pattern_entropy(binary_v,s*s, 1)
        if self.organ=='brain':
            binary_v = jnp.where(array > threshold, 1, 0)
            binary_v=binary_v.reshape(s, -1)
            binary_v=binary_v[:, self.transient_length:]
    
            entropy, normalised=self.pattern_entropy(binary_v,s, 1)
        return entropy, normalised
    
    def entropy(self, frame_size=9):
        """
        Splits the input array into smaller sequences of size (frame_size, T)
        and calculates the entropy for each sequence using the entropycalc function.
    
        Parameters:
            array (jnp.ndarray): Input array of shape (N, T).
            frame_size (int): Size of the smaller sequences. Default is 9.
        
        Returns:
            jnp.ndarray: Array of entropies for each sequence.
            jnp.ndarray: Array of normalized entropies for each sequence.
        """
        if self.organ == 'brain': 
            array=self.vs.T  
            N, T = array.shape
            num_frames = N // frame_size

            def calculate_entropy_for_frame(i):
                frame = lax.dynamic_slice(array, (i, 0), (frame_size, T))
                return self.handling_subsets(frame, frame_size, threshold= 0.5)

            indices = jnp.arange(0, N, frame_size)
    
            entropies, normalized_entropies = vmap(calculate_entropy_for_frame)(indices)
            self.entropy=jnp.mean(normalized_entropies)
        if self.organ == 'heart':
            N_x=self.N
            array=self.vs.T 
            frame_size=3
            # TODO : add option to remove boundary sites
            array=array.reshape(N_x, N_x, -1)
            N, _, T = array.shape
            num_frames = (N // frame_size) ** 2
            def calculate_entropy_for_frame(i, j):
    
                frame = lax.dynamic_slice(array, (i, j, 0), (frame_size, frame_size, T))
                return self.handling_subsets(frame, frame_size)
            indices = jnp.arange(0, N, frame_size)
    
            entropies, normalized_entropies = vmap( lambda i: vmap(lambda j: calculate_entropy_for_frame(i, j))(indices))(indices)
            self.entropy=jnp.mean(normalized_entropies)



class FHN_coherence:
    def __init__(self, model=None, vs=None, ws=None, ts=None, transient_length=100):
        if model is not None:
            self.ts = model.ts
            self.vs = model.vs
            self.ws = model.ws
            self.N = model.N
            self.block = model.block
            self.organ = model.organ
            self.transient_length=transient_length
        elif vs is not None and ws is not None:
            self.vs = vs
            self.ws = ws
            self.ts = ts
            self.N = vs.shape[1]
            self.transient_length=transient_length
        else:
            raise ValueError("Either model or vs and ws must be provided")


    def calculate_coherence(self,  V, window_size, step_size=1):
        """
        Calculate the coherence order parameter R_V(t) using a sliding window approach.
    
        Parameters:
            V (jax.numpy.DeviceArray): A (N, T) array where N is the number of neurons, and T is the number of
            window_size (int): Size of the sliding window (number of time steps).
            step_size (int): Step size for sliding the window (default is 1).
        
        Returns:
            average coherence
        """
        N, T = V.shape  # Number of neurons and time points
    
    # Function to calculate coherence for a single window
        def coherence_for_window(start_idx):
            V_window = lax.dynamic_slice(V, (0, start_idx), (N, window_size))  # Extract window
        
            # Step 1: Population mean at each time step
            V_bar = jnp.mean(V_window, axis=0)  # Shape: (window_size,)
        
            # Step 2: Numerator (variance of population mean over neurons)
            V_bar_squared_mean = jnp.mean(V_bar**2)  # ⟨V̄(t)²⟩
            V_bar_mean_squared = jnp.mean(V_bar)**2  # ⟨V̄(t)⟩²
            numerator = jnp.sqrt(V_bar_squared_mean - V_bar_mean_squared)
        
            # Step 3: Denominator (variance of individual neurons)
            V_squared_mean = jnp.mean(V_window**2, axis=1)  # ⟨Vᵢ(t)²⟩ over window
            V_mean_squared = jnp.mean(V_window, axis=1)**2  # ⟨Vᵢ(t)⟩² over window
            denominator = jnp.sqrt(jnp.mean(V_squared_mean - V_mean_squared))
        
            # Step 4: Coherence order parameter for the window
            return numerator / denominator
    
        # Calculate coherence for all windows using sliding indices
        window_starts = jnp.arange(0, T - window_size + 1, step_size)
        R_V_t = vmap(coherence_for_window)(window_starts)
    
        return jnp.mean(R_V_t)
    # calculates coherence for both organs
    def coherence(self, window_size=1000, step_size=1): #TODO : fix the problem with Tin

        v_values=self.vs.T
        if self.organ == 'brain':
            self.coherence=self.calculate_coherence(v_values[:, self.transient_length:], window_size, step_size)
        if self.organ == 'heart':
            N_x=self.N
            if N_x>8:
                block=~self.block[4:(N_x-4), 4:(N_x-4)]  
                v_values=v_values.reshape(N_x, N_x, -1)
                v_values=v_values[4:(N_x-4), 4:(N_x-4), :]

            
                #TODO: implement this in JAX. I tried several times and failed
                R=[]
                for j in range(N_x-8):  # Iterate over the correct dimension
                    filtered_column = v_values[:, j, self.transient_length:][block[:,j]]
                    R.append(self.calculate_coherence(filtered_column, 500))

                
            else:
                warnings.warn("Too few nodes to address boundary conditions")   
                block=~self.block
                v_values=v_values.reshape(N_x, N_x, -1)
                R=[]
                for j in range(N_x):
                    filtered_column = v_values[:, j, self.transient_length:][block[:,j]]      
                    R.append(self.calculate_coherence(filtered_column, 500))
            R=jnp.array(R)
            self.coherence=jnp.mean(R)
            



















































    
