# COARSE GRAINING VIA REPARAMETERIZATION CODE
# This file contains the imports and definitions that are used in the other files in the src directory.




import time

import torch
# import torch.nn.functional as F
import numpy as np

V = lambda x: x.detach().cpu().numpy()


# CG module:
# 1. compute the force constant matrix
# 2. compute the graph laplacian
# 3. compute the eigenvectors of the graph laplacian
# 4. pick the k eigenvectors with the lowest eigenvalues
# 5. compute the CG modes

import warnings
# warnings.filterwarnings('ignore')

class CoarseGrainer:
    def __init__(self, energy_func, num_cg_modes=None):
        """Defines Coarse-grained (CG) variables using the normal modes of the energy function.
        The normal modes are related to the force constant matrix of the particle system.
        The force constant matrix is the Hessian of the energy function with respect to the coordinates.
        The Hessian is a 4-tensor with shape (N, d, N, d).
        The first two indices correspond to the coordinates of the first atom, 
        and the second pair to the second atom. 
        
        We take the norm of the Hessian over the spatial coordinates of both atoms 
        to define the force constant matrix (symmetric of shape (N,N)) for the system. 
        The force constant for a bond is the curvature of the energy at the equilibrium bond length
        
        We consider the force constant matrix as the adjacency matrix of a graph.
        We define the graph laplacian as the diagonal matrix of the row sums of 
        the adjacency matrix minus the adjacency matrix.
        
        Then, to define the CG variables, we compute the eigenvectors of the graph laplacian.
        
        Args:
            energy (_type_): The energy function of the particle system. 
                It should take the coordinates as its first input, e.g. energy(x), and return a number.
            num_cg_modes (int, optional): the number of CG modes to compute. 
                If not specified, N/3 modes will be computed. Defaults to None.
        """
        self.energy = energy_func
        # define the Hessian of the energy function with respect to the coordinates
        self.get_hessian_func()
        self.num_cg_modes = num_cg_modes
        
    def get_hessian_func(self):
        self.hessian = torch.func.hessian(self.energy, argnums=0)

    # make the hessian norm over dims 1,3 a function
    def get_force_constant_matrix(self, x):
        """Compute the force constant matrix of the system"""
        # compute the Hessian
        H = self.hessian(x).norm(dim=(1,3))
        return H
    
    def get_sample_force_constant_matrix(self, x_samples):
        """Compute the force constant matrix of the system"""
        # make sure x_samples is a tensor with at least 3 dimensions
        assert len(x_samples.shape) >= 2
        if len(x_samples.shape) == 2:
            x_samples = x_samples.unsqueeze(0)
            
        # compute the Hessian for each ro in x_samples and take the mean over samples
        # H = torch.stack([self.get_force_constant_matrix(x) for x in x_samples]).mean(dim=0)
        # for memory efficiency, we can compute the Hessian for each row in x_samples and take the mean
        H = torch.zeros(x_samples.shape[-2], x_samples.shape[-2]).to(x_samples.device)
        for x in x_samples:
            H += self.get_force_constant_matrix(x)
        H /= x_samples.shape[0]
        return H
    
    def get_heat_capacity(self, x_samples):
        """Compute the heat capacity of the system"""
        H = self.get_sample_force_constant_matrix(x_samples)
        return H.trace()
    
    def get_laplacian(self, A):
        return torch.diag(A.sum(dim=1)) - A
    
    def get_cg_modes(self, x_samples, k=None, update_force_constant_matrix=True):
        if k is None:
            if self.num_cg_modes is None:
                k = np.ceil(x_samples.shape[-2]/3).astype(int)
                warnings.warn(f'k not specified. Using k={k}')
            else:
                k = self.num_cg_modes
        # initialize the force constant matrix, if not already initialized
        if not hasattr(self, 'force_constant_matrix'):
            self.force_constant_matrix = torch.zeros(x_samples.shape[-2], x_samples.shape[-2]).to(x_samples.device)
        # also, if update_force_constant_matrix is True, we will update the force constant matrix
        # if it is False, we will reinitialize the force constant matrix
        if not update_force_constant_matrix:
            self.force_constant_matrix = torch.zeros(x_samples.shape[-2], x_samples.shape[-2]).to(x_samples.device)
        # time to compute the CG modes
        t0 = time.time()
        # compute the force constant matrix
        self.force_constant_matrix += self.get_sample_force_constant_matrix(x_samples)
        print(f'force constant matrix computed in {time.time()-t0:.2f} seconds')
        t0 = time.time()
        # compute the spectrum of the Laplacian of the force constant matrix
        self.get_L_spectrum()
        print(f'spectrum of the Laplacian computed in {time.time()-t0:.2f} seconds')
        # pick the k eigenvectors with the lowest eigenvalues
        self.cg_idx = torch.argsort(self.L_eigenvalues.abs())[:k]
        # CG modes: get the k eigenvectors with the lowest eigenvalues
        self.cg_modes = self.L_eigenvectors[:, self.cg_idx]
        # also, store the eigenvalues coming from the force constant matrix
        # to do so, we will first find which eigenvectors correspond to the CG modes
        # then we will pick the eigenvalues corresponding to the CG modes
        # overlap = torch.einsum('ij,jk->ik', self.cg_modes.T, self.A_eigenvectors)
        # self.cg_eigenvalues = self.A_eigenvalues[overlap.argmax(dim=1)]
        # self.cg_eigenvalues = self.A_eigenvalues[self.cg_idx]
        # instead of eigenvalues, it would be more accurate to 
        # use the expectation value of the force constant matrix w.r.t. the CG modes
        self.cg_eigenvalues = (self.cg_modes.T @ self.force_constant_matrix @ self.cg_modes).diagonal()
        
    def get_L_spectrum(self):
        # compute the graph laplacian
        self.L = self.get_laplacian(self.force_constant_matrix)
        # compute the eigenvectors of the graph laplacian
        self.L_eigenvalues, self.L_eigenvectors = torch.linalg.eigh(self.L)
        # also compute spectrum of the force constant matrix (will use this for scaling the CG modes)
        self.A_eigenvalues, self.A_eigenvectors = torch.linalg.eigh(self.force_constant_matrix)
    
    def get_cg_variables(self, x):
        """compute the CG variables"""
        return self.cg_modes.T @ x
    
    def get_cg_energy(self, x):
        """compute the CG energy"""
        return self.energy(self.get_cg_variables(x))
    
    def get_cg_force_constant_matrix(self):
        """compute the CG force constant matrix"""
        return self.cg_modes.T @ self.force_constant_matrix @ self.cg_modes
    
    def get_cg_hessian(self, x):
        """compute the CG Hessian. Recall that the Hessian is shape (N,d,N,d)
        we want to compute the Hessian with respect to the CG variables
        the CG variables are shape (N,k)
        the CG Hessian is shape (k,d,k,d)"""
        # we can compute the CG Hessian by contracting the Hessian with the CG modes
        return torch.einsum('ijkl,im,kn->mjnl', self.hessian(x), self.cg_modes, self.cg_modes)
    
    def cg_reparametrize(self, x):
        """Reparametrize the coordinates using the CG variables"""
        z = self.get_cg_variables(x)
        # We want to define the CG variables z as the new optimization variables
        # we will detach z from the graph and set requires_grad=True
        # this will make z the new optimization variables
        return z.clone().detach().requires_grad_(True)
        
    def get_x_from_cg(self, z):
        """Get the coordinates from the CG variables"""
        return self.cg_modes @ z
    
    # Example usage of the CoarseGrainer class
    # cg = CoarseGrainer(energy_func=energy, num_cg_modes=7)
    # cg.get_cg_modes(x)
    # cg.cg_modes.shape
    
