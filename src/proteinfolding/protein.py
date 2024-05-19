import os

import torch

# import openmm as omm
from openmm import app

import requests

def download_pdb_file(pdb_id, dir = './'):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(dir, f'{pdb_id}.pdb')
        with open(file_path, 'w') as f:
            f.write(response.text)
        return file_path
    else:
        print(f'Failed to download PDB: {pdb_id}')

def get_file_text(name):
    with open(name, 'r') as f:
        s = f.readlines()
    return ''.join(s)

import py3Dmol

def pdb_3d_view(pdb_file, width=400, height=300):
    p_ = py3Dmol.view(data=get_file_text(pdb_file),width=width, height=height)
    p_.setStyle({'cartoon': {'color':'spectrum'}}); #'stick', 'cartoon'
    return p_

class Protein:
    def __init__(self, pdb_file):
        
        self.process_data_openmm(pdb_file)
        self.get_node_info()
        self.get_edge()
        self.get_adjacency_matrix()
        self.get_distance_matrix()
        self.get_random_distance_matrix()
        self.get_harmonic_bond_forces()
        self.get_harmonic_angle_forces()
        self.get_torsion_angle_forces()
        self.get_nonBonded_forces()

    def process_data_openmm(self, pdb_file):
        # check if the pdb_data is a path to a file that exists
        # if it doesn't exist, download the pdb file from the rcsb website
        pdb = app.PDBFile(pdb_file)
            
        self.modeller = app.Modeller(pdb.topology, pdb.positions)
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        self.modeller.addHydrogens(forcefield)
        self.modeller.deleteWater()
        #self.modeller.addSolvent(forcefield, model="tip3p", padding=.05 * unit.nanometer)
        self.system = forcefield.createSystem(self.modeller.topology)#, nonbondedMethod=PME, constraints=HBonds)


    def get_node_info(self):
        self.node_positions = torch.tensor([[i.x, i.y, i.z] for i in self.modeller.getPositions()])
        self.node_label =  [i.element.symbol for i in self.modeller.getTopology().atoms()]
        self.residue = [i.residue.name for i in self.modeller.getTopology().atoms()]

    def get_edge(self):
        self.edge_list = torch.tensor([(bond.atom1.index, bond.atom2.index) for bond in self.modeller.getTopology().bonds()])

    def get_adjacency_matrix(self):
        self.adjacency_matrix = torch.zeros((len(self.node_positions), len(self.node_positions)))

        for i in self.edge_list:
            self.adjacency_matrix[i[0]][i[1]] = 1
            self.adjacency_matrix[i[1]][i[0]] = 1

    def get_distance_matrix(self):
        self.distance_matrix = (self.node_positions[None]- self.node_positions[:,None]).norm(dim=-1)

    def get_random_distance_matrix(self):
        random_positions = torch.rand(len(self.node_positions),3) * torch.amax(self.node_positions)
        self.random_distance_matrix = (random_positions[None]- random_positions[:,None]).norm(dim = -1)

    def get_spectrum():
        None

    def get_harmonic_bond_forces(self):
        # id1, id2, equilibrium length, k
        # self.harmonic_bond_data = []
        force = self.system.getForces()[0]
        n = force.getNumBonds()
        # make dict of ids and their corresponding equilibrium length and k
        # ids: nx2 tensor of ids (int)
        # equilibrium length: nx1 tensor of equilibrium length (float)
        # k: nx1 tensor of k (float)
        self.harmonic_bond_data = dict(
            ids = torch.zeros((n,2), dtype = torch.int),
            a = torch.zeros(n),
            k = torch.zeros(n)
        )
        for i in range(n): #the 0 id means the harmonic bond force
            d = force.getBondParameters(i)
            # self.harmonic_bond_data += [(d[0],d[1], d[2]._value, d[3]._value)]
            self.harmonic_bond_data['ids'][i] = torch.tensor([d[0],d[1]])
            self.harmonic_bond_data['a'][i] = d[2]._value
            self.harmonic_bond_data['k'][i] = d[3]._value
        # self.harmonic_bond_data = torch.tensor(self.harmonic_bond_data)

    def get_harmonic_angle_forces(self):
        # id1, id2, id3, equilibrium angle, k
        # self.harmonic_angle_data = []
        force = self.system.getForces()[4]
        n = force.getNumAngles()
        self.harmonic_angle_data = dict(
            ids = torch.zeros((n,3), dtype = torch.int),
            t = torch.zeros(n),
            k = torch.zeros(n)
        )
        for i in range(n): #the 4 id means the harmonic angle force
            d = force.getAngleParameters(i)
            # self.harmonic_angle_data += [(d[0],d[1],d[2], d[3]._value, d[4]._value)]  # d[3]._value*180/pi
            self.harmonic_angle_data['ids'][i] = torch.tensor([d[0],d[1],d[2]])
            self.harmonic_angle_data['t'][i] = d[3]._value
            self.harmonic_angle_data['k'][i] = d[4]._value
        # self.harmonic_angle_data = torch.tensor(self.harmonic_angle_data)

    def get_torsion_angle_forces(self):
        # id1, id2, id3, id4, n, equilibrium torsion angle, k
        # self.torsion_angle_data = []
        force = self.system.getForces()[2]
        n = force.getNumTorsions()
        self.torsion_angle_data = dict(
            ids = torch.zeros((n,4), dtype = torch.int),
            n = torch.zeros(n, dtype = torch.int),
            t = torch.zeros(n),
            k = torch.zeros(n)
        )
        for i in range(n): #the 2 id means the torsion angle force
            d = force.getTorsionParameters(i)
            # self.torsion_angle_data += [(d[0],d[1],d[2], d[3], d[4], d[5]._value, d[6]._value)]
            self.torsion_angle_data['ids'][i] = torch.tensor([d[0],d[1],d[2],d[3]])
            self.torsion_angle_data['n'][i] = d[4]
            self.torsion_angle_data['t'][i] = d[5]._value
            self.torsion_angle_data['k'][i] = d[6]._value
            
        # self.torsion_angle_data = torch.tensor(self.torsion_angle_data)


    def get_nonBonded_forces(self):
        # charge, sigma, epsilon
        force = self.system.getForces()[1]
        n = force.getNumParticles()
        self.non_bonded_data = dict(
            charge = torch.zeros(n),
            sigma = torch.zeros(n),
            epsilon = torch.zeros(n)
        )
        for i in range(n): #the 1 id non bonded force
            d = force.getParticleParameters(i)
            # self.non_bonded_data += [(d[0]._value, d[1]._value, d[2]._value)]
            self.non_bonded_data['charge'][i] = d[0]._value
            self.non_bonded_data['sigma'][i] = d[1]._value
            self.non_bonded_data['epsilon'][i] = d[2]._value

        # self.non_bonded_data = torch.tensor(self.non_bonded_data)