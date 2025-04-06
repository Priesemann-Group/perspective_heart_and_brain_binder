import networkx as nx
import numpy as np 
from scipy.sparse import diags, coo_matrix
from scipy import sparse
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import sparse
from jax import lax, vmap


class FHN_model:
    def __init__(self,
                organ='brain',
                N='organ_default', 
                a='organ_default', 
                b='organ_default', 
                e='organ_default', 
                sigma='organ_default', 
                Dv='organ_default', 
                v0='zeros', 
                w0='zeros',
                m='organ_default',
                p='organ_default',
                k='organ_default',
                random_key=jr.PRNGKey(1000), 
                v0_sigma=None, 
                w0_sigma=None,
                adjacency_seed=1000,
                stimulus_period=2000):
   
        if type(organ)==str and organ == 'brain':
            if type(N)==str and N == 'organ_default':
                N = 1000
            if type(a)==str and a == 'organ_default':
                a = 3
            if type(b)==str and b == 'organ_default':
                b = 0.05
            if type(e)==str and e == 'organ_default':
                e = 0.01
            if type(sigma)==str and sigma == 'organ_default':
                sigma = 0.008
            if type(Dv)==str and Dv == 'organ_default':
                Dv = 1
            elif Dv != 1:
                print('A Dv value other than 1 is not allowed for a brain simulation as the parameter is absorbed in m. Dv will be set to 1 instead.')
                Dv = 1
            if type(m)==str and m == 'organ_default':
                m = 0.005
            if type(p)==str and p == 'organ_default':
                p = 0    # has no influence
            if type(k)==str and k == 'organ_default':
                k = 8   # has no influence

        elif type(organ)==str and organ == 'heart':
            if type(N)==str and N == 'organ_default':
                N = 100
            if type(a)==str and a == 'organ_default':
                a = 3
            if type(b)==str and b == 'organ_default':
                b = 0.05
            if type(e)==str and e == 'organ_default':
                e = 0.01
            if type(sigma)==str and sigma == 'organ_default':
                sigma = 0.0001
            if type(Dv)==str and Dv == 'organ_default':
                Dv = -0.04
            if type(m)==str and m == 'organ_default':
                m = 0.005  # has no influence
            if type(p)==str and p == 'organ_default':
                p = 0
            if type(k)==str and k == 'organ_default':
                k = 8      # has no influence
        else:
            raise ValueError('Organ must be either brain or heart')

        # Save model parameters
        self.organ=organ
        self.adjacency_seed=adjacency_seed
        self.stimulus_period=stimulus_period

        self.N = N
        self.a = a
        self.b = b
        self.e = e
        self.sigma = sigma
        self.Dv=Dv
        self.m=m
        self.p=p
        self.k=k

        ## Initiate the coupling matrix and initial conditions
        if type(organ)==str and organ == 'brain':
            # Coupling matrix
            self.initiate_random_graph(seed=adjacency_seed)
            self.block = None # Blocked, fibrotic cells only exist in heart simulations

            # Initial conditions
            if type(v0) == str and v0=='zeros':
                v0 = jnp.zeros(N)
            if type(w0) == str and w0=='zeros':
                w0 = jnp.zeros(N)
        
            if type(v0) == str and v0=='random':
                v0 = jax.random.normal(random_key, shape=(2*N,))*v0_sigma
            if type(w0) == str and w0=='random':
                v0 = jax.random.normal(random_key, shape=(2*N,))*w0_sigma
        
        elif type(organ)==str and organ == 'heart':
            # Coupling matrix
            self.generate_laplacian(seed=adjacency_seed)

            # Initial conditions
            indices = jnp.where((jnp.arange(N*N) % N == 0) & (self.block.flatten() == 0))[0]
            y0 = jnp.zeros(2 * N*N, dtype=jnp.float32)
            y0 = y0.at[indices].set(0.1)
            v0=y0[:N*N]
            w0=y0[N*N:]
        
        # Save initial conditions
        self.v0 = v0
        self.w0 = w0
        
        # Initialize solution obects
        self.ts = None
        self.vs = None
        self.ws = None

        params = {
            'organ': organ,
            'N': N,
            'a': a,
            'b': b,
            'e': e,
            'sigma': sigma,
            'adjacency_seed': adjacency_seed
        }

        if organ == 'brain':
            params['m'] = m
            params['k'] = k
        elif organ == 'heart':
            params['p'] = p
            params['Dv'] = Dv
            params['stimulus_period'] = stimulus_period
        print(params)
        

    def nullclines(self, v_array):
        return (v_array, self.a*v_array*(v_array-self.b)*(1-v_array))  # return the corresponding w_arrays
    
    
    def initiate_random_graph(self,  seed=100):
        '''
        Creates a random, directed, and weighted Erdos Renyi graph.
        Parameters:
            N: number of nodes
            k: mean nodal degree
            J: weight parameters. If homogeneous weights: constant float, if gaussian weigts: J=(J_mean, J_sigma)
            seed: seed for the ER graph generation
            weights: Type of weights, 'homogeneous' or 'gaussian'
            generator: random generator for random weights
        Returns:
            sparse jax.experimental coupling matrix 
        '''
        
        p = self.k / (self.N - 1)
        J=self.m/self.k
        # Create ER graph
        seed=self.adjacency_seed
        G = nx.erdos_renyi_graph(self.N, p, directed=True, seed=seed)
        
        # Put weights
        for u, v in G.edges():
            G[u][v]['weight'] = J
        
        # Get the adjacency matrix in sparse format
        
        adj_matrix = nx.to_scipy_sparse_array(G, weight='weight').tocsr()


        self.J = sparse.BCSR.from_scipy_sparse(adj_matrix)
    
    #Laplacian generation for the heart
    def generate_laplacian(self, sparse_matrix=True, seed=101):
        #TODO: implement fully in jax
        num_nodes = self.N * self.N
        adj_rows = []
        adj_cols = []
        adj_data = []
        seed= self.adjacency_seed
        # Generate random conDvction blocks
        np.random.seed(seed)
        conduction_blocks = np.random.rand(self.N, self.N) < self.p

        # Function to map grid (i, j) to a single node index
        def node_index(i, j):
            return i * self.N + j

        # Define neighbors for the nine-point stencil with weights
        neighbors = [
            (-1, 0, .5),     # up
            (1, 0, .5),      # down
            (0, -1, .5),     # left
            (0, 1, .5),      # right
            (-1, -1, .25),   # top-left
            (-1, 1, .25),    # top-right
            (1, -1, .25),    # bottom-left
            (1, 1, .25)      # bottom-right
        ]
    
        # Build adjacency structure excluding conDvction blocks
        indices = np.array([[i, j] for i in range(self.N) for j in range(self.N)])
        idx = node_index(indices[:, 0], indices[:, 1])

        for di, dj, weight in neighbors:
            ni = indices[:, 0] + di
            nj = indices[:, 1] + dj

        # Step 1: Filter for in-bounds neighbors
            in_bounds = (ni >= 0) & (ni < self.N) & (nj >= 0) & (nj < self.N)
    
        # Step 2: Find valid indices (in-bounds) to avoid shape mismatches
            valid_indices = np.where(in_bounds)[0]
            ni_valid = ni[valid_indices]
            nj_valid = nj[valid_indices]

        # Step 3: Apply conDvction block exclusion on the filtered indices
            valid_conduction = ~conduction_blocks[ni_valid, nj_valid]
            valid_node = ~conduction_blocks[indices[valid_indices, 0], indices[valid_indices, 1]]
            valid = valid_conduction & valid_node

        # Step 4: Append data for fully valid connections
            adj_rows.extend(idx[valid_indices][valid])
            adj_cols.extend(node_index(ni_valid[valid], nj_valid[valid]))
            adj_data.extend([weight] * int(np.sum(valid)))


        # Create adjacency and degree matrices
        adj_matrix = coo_matrix((adj_data, (adj_rows, adj_cols)), shape=(num_nodes, num_nodes))
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        degree_matrix = diags(degrees)

        # Construct Laplacian matrix
        laplacian_matrix = degree_matrix - adj_matrix

        if sparse_matrix:

            self.J= sparse.BCSR.from_scipy_sparse(laplacian_matrix)
            self.block= jnp.array(conduction_blocks)
    
        else:
            self.J= laplacian_matrix.todense()
            self.block= conduction_blocks

    # Deterministic part of the differential equation
    def FHN_graph(self, v, w):
        dv = self.a*v*(v-self.b)*(1-v) + self.Dv*(self.J @ v) - w 
        dw = self.e*(v-w)

        return (dv,dw)
    
    # Stochastic part of the differential equation
    def FHN_graph_noise(self):
        noise = self.sigma*jnp.ones(self.N)
        return noise
    def run_simulation(self, delta_t=0.1, T=3000.0, n_stored_states=3000, random_key=jr.PRNGKey(1000)): 
              
        # Calculate the number of solver steps based on the total time and delta_t
        num_steps = int(T / delta_t)
        output_every = int(max(num_steps/n_stored_states,1))
        
        Ntot = self.N
        
        # Heart-specific preparations 
        if self.organ=='heart':
            # meaning of N changes
            Ntot= self.N*self.N
            # note indices fibrotic conDvction blocks, needed for stimulus
            indices = jnp.where((jnp.arange(Ntot) % self.N == 0) & (self.block.flatten() == 0))[0]
                
        # Initialize output arrays
        vs = jnp.zeros((n_stored_states, Ntot))
        ws = jnp.zeros((n_stored_states, Ntot))

        # Define the scan function
        def scan_fn(step, carry):
            v, w, key, vs, ws = carry
            key, subkey = jr.split(key)

            # Update variables
            deterministic_update = self.FHN_graph(v, w)
            noise_update = jr.normal(subkey, v.shape) * self.sigma
            v = v + deterministic_update[0]*delta_t +  jnp.sqrt(delta_t) * noise_update
            w = w + deterministic_update[1]*delta_t

            # Heart-specific: periodic stimulus from SA nodes
            if self.organ=='heart':
                # Apply stimulus to the specified indices
                v = jax.lax.cond((step > 0) & (step % int(self.stimulus_period / delta_t) == 0),
                        lambda v: v.at[indices].add(0.1),
                        lambda v: v,
                        v)
            
            vs = vs.at[step//output_every,:].set(v)
            ws = ws.at[step//output_every,:].set(w)
            return (v, w, key, vs, ws)

        # Run the scan function
        _, _, _, vs, ws = jax.lax.fori_loop(0, num_steps, scan_fn, (self.v0, self.w0, random_key, vs, ws))
        
        # Make sure only at most n_stored_states many time points are stored
        self.ts = jnp.linspace(0, T, n_stored_states)
        self.vs = vs
        self.ws = ws

        return None
