from numpy import pi

import torch
from openmm import unit

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coarsegrainer.energy import EnergyModule
# from ..coarsegrainer.energy import EnergyModule

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

def Coulomb_repulsion(r, r0=1, p=1, eps = EPSILON):
    s = r/r0
    return 1/(s**p + eps )

class EnergyProtein(EnergyModule):
    Q_ELEC = unit.elementary_charge.conversion_factor_to(unit.coulomb)
    COULOMB_CONST = 0.001*((Q_ELEC**2)/4/pi/8.85e-21) #kJ
    def __init__(self, prot, device = 'cpu'):
        self.prot = prot
        self.device = device
        self.prep_data()
    
    def prep_data(self):
        self.bond_data = self.prot.harmonic_bond_data.copy()
        self.angle_data = self.prot.harmonic_angle_data.copy()
        self.torsion_data = self.prot.torsion_angle_data.copy()
        self.non_bond_data = self.prot.non_bonded_data.copy()
        # each of the data is a dictionary with keys ids and other variables
        # each value is a torch tensor
        # we should convert the values to the device
        # 1. iterate over all four dictionaries
        # 2. iterate over all keys
        # 3. convert the values to the device
        for data in [self.bond_data, self.angle_data, self.torsion_data, self.non_bond_data]:
            for key in data.keys():
                data[key] = data[key].to(self.device)
                
        # non-bonded data prep 
        sigma = self.non_bond_data['sigma']
        epsilon = self.non_bond_data['epsilon']
        charge = self.non_bond_data['charge']
        
        self.LJ_sigma = (sigma[None] + sigma[:,None])/2
        self.LJ_epsilon = torch.sqrt(epsilon[None]* epsilon[:,None])
        self.charge2 = charge[None] * charge[:,None]
        # to avoid self-interaction and double counting, only use the upper triangle
        # self.LJ_sigma = torch.triu(self.LJ_sigma, diagonal = 1)
        self.LJ_epsilon = torch.triu(self.LJ_epsilon, diagonal = 1)
        self.charge2 = torch.triu(self.charge2, diagonal = 1)
        

    def harmonic_bond_energy(self, x,):
        ids = self.bond_data['ids']
        a = self.bond_data['a']
        k = self.bond_data['k']
        
        distance  = (x[ids[:,0]] - x[ids[:,1]]).norm(dim=-1)
        energy_bond = (k @ ((distance - a)**2))/2
        return energy_bond

    def harmonic_angle_energy(self, x):
        ids = self.angle_data['ids']
        t = self.angle_data['t']
        k = self.angle_data['k']

        vector1 = x[ids[:,0]] - x[ids[:,1]]
        vector2 = x[ids[:,2]] - x[ids[:,1]]

        unit_vector1 = vector1/vector1.norm(dim=1, keepdim=True)
        unit_vector2 = vector2/vector2.norm(dim=1, keepdim=True)
        
        angle = torch.arccos((unit_vector1 * unit_vector2).sum(dim = 1))

        energy_angle = (k @ ((angle - t)**2))/2

        return energy_angle


    def torsion_angle_energy(self, x):
        ids = self.torsion_data['ids']
        n = self.torsion_data['n']
        t = self.torsion_data['t']
        k = self.torsion_data['k']
        
        # 0,1,2,3: i,j,k,l
        # i-j
        vector1 = x[ids[:,0]] - x[ids[:,1]]
        # j-k
        vector2 = x[ids[:,2]] - x[ids[:,1]]
        # k-l
        vector3 = x[ids[:,2]] - x[ids[:,3]]
        
        # crx1 = (i-j) x (j-k)
        cross1 = torch.cross(vector1, vector2, dim=1)
        # crx2 = (j-k) x (k-l)
        cross2 = torch.cross(vector2, vector3, dim=1)
        
        m1 = torch.cross(cross1, vector2, dim=1)
        m2 = torch.cross(cross2, vector2, dim=1)
        
        dot = (m1 * m2).norm(dim=-1)
        
        cross = torch.cross(m1, m2, dim=1)
        
        # Calculate the sign of the angle
        sign = torch.sign((vector2 * cross).norm(dim=-1))
        # # Calculate the sign of the angle
        # # sign = torch.sign(torch.sum(vector2 * cross, dim=-1))
        # Calculate the angle
        angle = sign * torch.arctan2((cross).norm(dim = -1), dot)

        energy_angle = k @ (1 + torch.cos(n * angle - t))
        
        return energy_angle

    def torsion_angle_energy2(self, x):
        """ Compute energy of torsion angles. 
        The formula for torsion angle energy is:
        E = k * (1 + cos(n * angle - t))
        where angle is the torsion angle, n is the multiplicity, t is the equilibrium angle, and k is the force constant.
        Given four atoms i, j, k, l, the torsion angle is the angle between the planes defined by the atoms i,j,k and j,k,l.
        To compute the torsion angle, we first compute the cross product of the vectors i-j and j-k, and the cross product of the vectors j-k and k-l.
        Then we compute the cross product of the two cross products, and the angle is the arctan of the norm of this cross product divided by the dot product of the two cross products.
        Finally, we compute the energy using the formula above.
        
        crx1 = (i-j) x (j-k)
        crx2 = (j-k) x (k-l)
        crx = crx1 x crx2
        dot = crx1 . crx2
        angle = arctan(norm(crx)/dot)
        energy = k * (1 + cos(n * angle - t))
        
        x: torch tensor of shape (n,3) with coordinates
        return: torch tensor of shape (m,) with energy of each torsion angle
        """
        ids = self.torsion_data['ids']
        n = self.torsion_data['n']
        t = self.torsion_data['t']
        k = self.torsion_data['k']
        
        # 0,1,2,3: i,j,k,l
        # i-j
        vector1 = x[ids[:,0]] - x[ids[:,1]]
        # j-k
        vector2 = x[ids[:,1]] - x[ids[:,2]]
        # k-l
        vector3 = x[ids[:,2]] - x[ids[:,3]]
        
        # crx1 = (i-j) x (j-k)
        cross1 = torch.cross(vector1, vector2, dim=1)
        # crx2 = (j-k) x (k-l)
        cross2 = torch.cross(vector2, vector3, dim=1)
        
        # m1 = torch.cross(cross1, vector2, dim=1)
        # m2 = torch.cross(cross2, vector2, dim=1)
        # dot = (m1 * m2).norm(dim=-1)
        # cross = torch.cross(m1, m2, dim=1)
        
        # crx = crx1 x crx2
        cross = torch.cross(cross1, cross2, dim=1)
        # dot = crx1 . crx2
        dot = (cross1 * cross2).sum(dim=-1)

        # # Calculate the sign of the angle
        # sign = torch.sign((vector2 * cross).norm(dim=-1))
        # # Calculate the sign of the angle
        # # sign = torch.sign(torch.sum(vector2 * cross, dim=-1))

        # # Calculate the angle
        # angle = sign * torch.arctan2((cross).norm(dim = -1), dot)
        
        angle = torch.atan2((cross).norm(dim = -1), dot)

        energy_angle = k @ (1 + torch.cos(n * angle - t))
        
        return energy_angle

    def vdw_energy(self, distance):
        # distance = (x[None]- x[:,None]).norm(dim=-1)
        return (4*self.LJ_epsilon*LJ_potential(distance/self.LJ_sigma)).sum()
    
    def coulomb_energy(self, distance):
        # distance = (x[None]- x[:,None]).norm(dim=-1)
        return self.COULOMB_CONST*(self.charge2*Coulomb_repulsion(distance)).sum()

    def non_bonded_energy(self, x):
        distance = (x[None]- x[:,None]).norm(dim=-1)
        return self.vdw_energy(distance) + self.coulomb_energy(distance)
        
    def energy(self, x):
        return self.harmonic_bond_energy(x) \
            + self.harmonic_angle_energy(x) \
            + self.torsion_angle_energy(x) \
            + self.non_bonded_energy(x)

