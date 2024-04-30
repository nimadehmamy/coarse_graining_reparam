    
#### Energy function
import numpy as np
import torch


V = lambda x: x.detach().cpu().numpy()

Laplacian = lambda A: torch.diag(A.sum(-1))-A

# choose a small epsilon to avoid division by zero, 
# but not too small to avoid numerical instability
EPSILON = 1e-2

def LJ_potential(r,r0=1, p=6, eps = EPSILON):
    """pairwise Lennard-Jones potential
    V = (r/r0)^-2p - (r/r0)^-p
    r: array of distances
    eps: to cap max potential near r=0
    
    return: same shape as r
    """
    # s = torch.norm(r/r0, dim =-1)
    s = r/r0
    return 1/(s**(2*p) + eps )  - 1/( s**p + p*eps ) 

def repulsion(r,r0=1, p=6, eps = EPSILON):
    s = r/r0
    return 1/(s**p + eps ) 

####

def quadratic_potential(r,r0=1.):
    """
    (r/r0 - 1)**2
    r: array of distances 
    r0: (float) minimum 
    
    return: same shape as r  
    """
    # s = torch.norm(r/r0, dim =-1)
    return (r/r0-1)**2 


distance_matrix = lambda x: torch.norm(x[:,None]-x[None],dim=-1)

distance_matrix_idx = lambda x,idx: torch.norm(x[idx[0]]-x[idx[1]],dim=-1)


####

def get_rand_idx(n,k):
    # get k random index pairs
    # use torch.combinations to get all pairs of indices
    all_idx = torch.combinations(torch.arange(n),2).t()
    
    # get k random index pairs
    idx_rand = all_idx[:, torch.randperm(all_idx.shape[1])[:k]]
    # i = torch.randperm(n**2)[:k]
    # idx_rand = torch.concat(((i// n)[:,None], (i % n)[:,None]), dim=1)
    # because our forces depend on the norm |r_ij|, we don't need to keep both (i,j) and (j,i)
    # idx_rand = torch.concat((idx_rand, idx_rand[:,[1,0]]))
    return tuple(idx_rand)

# k = n**2//2 # 
# k = int(n* np.log(n))
# idx_rand = get_rand_idx(n,k) 

def spectrum_abs(L):
    """assuming Hermitian"""
    v,p = torch.linalg.eigh(L)
    # ensure they're sorted from small to large v
    i = torch.argsort(v.abs()) # may not be PSD
    v = v[i]
    p = p[:,i]
    return v,p


# we want an energy function which uses sparse coupling matrices. 
# The energy function should also take pair indices as input, to compute all-to-all energies.

# For efficiency, we'll define the energy function as a sum of indexed energy functions
# the terms in the energy will receive r directly, to avoid recomputing it.
# additionally, pairwise terms will receive the indices of the atoms involved.


# indexed distance matrix
def distmat_idx(x, idx_pairs):
    return torch.norm(x[idx_pairs[0]] - x[idx_pairs[1]], dim=-1)

# we will use LJ for van der Waals interactions. 
# the LJ potential is a function of the distance matrix r_ij
# it should also take the vdw radii of the atoms involved
def lj(r, m, n):
    return (r**(-m) - 2*r**(-n))


# # define sparse bond energy function
# def bond_energy_sparse(r, bond_matrix_sparse):
#     # the bond_matrix_sparse is a sparse matrix with the bond lengths at index i,j
#     # r is the distance matrix
#     return torch.sparse.mm(C_sparse, r)
    
#### Base energy class

# we define EnergyModule for converting an arbitrary energy function to the form we need
# The base model is just a wrapper around the energy function,
# and defines an update method to update the parameters of the energy function.

class EnergyModule:
    def __init__(self, energy_func, **energy_kws):
        """Base class for energy functions.
        
        Args:
            energy_func (callable): Energy function.
        """
        self.get_energy = energy_func
        self.energy_kws = energy_kws
        
    def update(self, **kws):
        """This method is used to update the parameters of the energy function.
        This may include defining pairs of atoms to compute the energy for, etc.
        """
        # self.energy_kws.update(kws)
        pass 
        
    def __call__(self, x):
        return self.get_energy(x, **self.energy_kws)
    
    def __repr__(self):
        return f'EnergyModule object with energy function {self.energy_func.__name__}'

#### Energy class 

# Define energy class instead of energy functions, to allow for updating indices, etc.
# We will have one base class, and then subclasses for different types of energies. 
# The base class will take a list of coupling matrices (A, A2, A_LJ, etc.) 
# and a list of energy functions (LJ, quadratic, etc.)
# it will have a method to process the coupling matrices (i.e. move them to device, threshold them, etc.),
# a method get the indices for the energy functions, and a method to compute the energy 
# associated with the coupling matrices and the energy functions.
# In addition to the couplings provided, it will also have a method to compute the repulsion energy
# for negative pairs where A_LJ=0 or A2=0.
# Finally, it will have a method to sum all the energies and return the total energy.

# since each energy function will have its own indices, we will have to keep track of them separately
# each energy function may also have its own parameters, which we will keep track of separately as well.
# it is easier if each energy function is a class, so that we
# can keep track of its parameters and indices in the same place.



class Energy:
    def __init__(self, A_list, energy_func_list, device='auto', num_neg_pairs='auto', thres_A=1e-4,
                log_name='Energy' ,**kws):
        """Class to compute the energy of a particle system using a list of coupling matrices and energy functions.
        The energy is computed as the sum of the energies of the energy functions,
        each weighted by the coupling matrices.
        The energy also includes a repulsion term for negative pairs where all the coupling matrices are zero.
        
        Args:
            A_list (list of torch.Tensor): List of coupling matrices.
            energy_func_list (_type_): List of energy functions.
            device (str, optional): Device for torch. If 'auto', 
                it will be 'cuda' if available, else 'cpu'.
                Defaults to 'auto'.
            thres_A (float, optional): Threshold for the coupling matrices. 
                Below this threshold, the coupling is ignored. Defaults to 1e-4.
        kwargs:
            rad2 (int, optional): _description_. Defaults to 1.
            radius_LJ (int, optional): _description_. Defaults to 1.
            lj_pow (int, optional): _description_. Defaults to 6.
            repel_pow (int, optional): _description_. Defaults to 4.
            repel_mag (float, optional): Magnitude of the repulsion energy. Defaults to 1e-2.
            num_neg_pairs (str, optional): Number of negative pairs to consider for repulsion.
                If 'auto', it will be n*log(n). Defaults to 'auto'.
        """
        self.device = (device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.thres_A = thres_A
        self.log_name = log_name
        self.process_kwargs(kws)
        self.process_coupling_matrices(A_list, **kws)
        self.energy_func_list = energy_func_list
        self.get_num_neg_pairs(num_neg_pairs)        
        self.get_indices()
        
    def process_coupling_matrices(self, A_list, **kws):
        self.A_list = [A.to(self.device) for A in A_list]
        
    def process_energy_funcs(self, energy_func_list, **kws):
        self.energy_func_list = energy_func_list
        
    def process_kwargs(self, kws):
        self.kws = kws
        self.rad2 = kws.get('rad2', 1)
        self.radius_LJ = kws.get('radius_LJ', 1)
        self.lj_pow = kws.get('lj_pow', 6)
        self.repel_pow = kws.get('repel_pow', 4)
        self.repel_mag = kws.get('repel_mag', 1e-2)
        
    def get_num_neg_pairs(self, num_neg_pairs):
        if num_neg_pairs == 'auto':
            n = len(self.A_list[0])
            num_neg_pairs = int(n*np.log(n))
        self.num_neg_pairs = num_neg_pairs
        
    def get_indices(self):
        # make a list of indices for each A in A_list
        self.indices = [torch.where(A.abs() > self.thres_A) for A in self.A_list]
        # find all negative pairs where all A in A_list are zero
        # this will be used for computing the repulsion energy
        self.get_all_neg_pairs()
        # we won't use all negative pairs for computing the energy, only num_neg_pairs of them
        self.update_neg_pairs()
        
    def get_all_neg_pairs(self):
        # make a list of negative indices for each A in A_list
        idx_inv = [torch.where( (A.abs()+torch.eye(len(A), device=self.device)) < self.thres_A) for A in self.A_list]
        # convert the list of tuples of indices into a list of tensors (for concatenation)
        idx_inv = [torch.stack(i,) for i in idx_inv]
        # concatenate the indices for all A in A_list
        idx_inv = torch.cat(idx_inv, dim=-1)
        # remove duplicates
        idx_inv = torch.unique(idx_inv, dim=-1)
        # keep all negative pairs for updating random pairs
        self.indices_neg_all = idx_inv
        
    def update(self):
        # update the coupling matrices
        # and the negative pairs
        self.update_neg_pairs()
        
    def update_neg_pairs(self):
        # choose num_neg_pairs random negative pairs from idx_inv randomly
        i = torch.randperm(self.indices_neg_all.shape[1])[:self.num_neg_pairs]
        self.indices_neg = (self.indices_neg_all[0][i], self.indices_neg_all[1][i])
        # self.indices_neg = (idx_inv[0][:self.num_neg_pairs], idx_inv[1][:self.num_neg_pairs])
        
        
    def get_energy(self, x):
        # for each A and energy function, compute the energy
        # as A_ij * energy_func(r_ij), where r_ij is the distance matrix
        # using the indices for each A
        # and sum all the energies
        energy = 0
        for A, idx, energy_func in zip(self.A_list, self.indices, self.energy_func_list):
            A_ij = A[idx]
            r_ij = distance_matrix_idx(x, idx)
            energy += torch.mean(A_ij * energy_func(r_ij))
            
        # compute the repulsion energy
        # for negative pairs where all A in A_list are zero
        # using the negative indices
        r_inv = distance_matrix_idx(x, self.indices_neg)
        energy_repel = torch.mean(repulsion(r_inv, p=self.repel_pow))
        energy += self.repel_mag * energy_repel
        
        return energy
        
    def __call__(self, x):
        return self.get_energy(x)
        
    def __repr__(self):
        return f'{self.log_name} object with {len(self.A_list)} coupling matrices and {len(self.energy_func_list)} energy functions'
    

