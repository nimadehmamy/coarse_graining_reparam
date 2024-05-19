from openmm.app import PDBFile, ForceField, Modeller, Simulation, PME, HBonds, DCDReporter
# Add reporters to save the simulation data
from openmm.app import PDBReporter, StateDataReporter
from sys import stdout
from openmm import LangevinIntegrator, unit
import time
import os

class ProteinFolderOMM:
    def __init__(self, pdb_file, forcefield='amber14-all.xml', water_model='amber14/tip3pfb.xml', 
            padding=1.0, temperature=298, timestep=2):
        """Protein folder using OpenMM
        Args:
            pdb_file (str): path to the pdb file
            forcefield (str): name of the forcefield to use
            water_model (str): name of the water model to use
            padding (float): padding to add to the box
            temperature (K, float): temperature of the simulation
            timestep (femto sec, float): timestep of the simulation
        """
        self.pdb_file = pdb_file
        self.get_trajectory_file(pdb_file)
        self.forcefield = forcefield
        self.water_model = water_model
        self.temperature = temperature*unit.kelvin
        self.timestep = timestep*unit.femtoseconds
        self.padding = padding
        self.history = {'total_energy': [], 'potential_energy': [], 'kinetic_energy': [], 'temperature': [],'time': []}
        self.setup()
        
    def get_trajectory_file(self, pdb_file):
        pdb_dir = os.path.dirname(pdb_file)
        pdb_id = os.path.basename(pdb_file).split('.')[0]
        self.trajectory_file = os.path.join(pdb_dir, f'refold_trajectory_{pdb_id}.pdb')
        
    def setup(self):
        # Load the PDB file
        self.pdb = PDBFile(self.pdb_file)

        # Set up the force field
        forcefield = ForceField(self.forcefield, self.water_model)

        # Create a modeller object and add solvent
        modeller = Modeller(self.pdb.topology, self.pdb.positions)
        modeller.addSolvent(forcefield, model='tip3p', padding=self.padding*unit.nanometers)
        # also create barebone modeller without solvent
        self.modeller_no_solvent = Modeller(self.pdb.topology, self.pdb.positions) 

        # Create the system
        system = forcefield.createSystem(modeller.topology,
                                        nonbondedMethod=PME,
                                        nonbondedCutoff=1.0*unit.nanometers,
                                        constraints=HBonds)

        # Define the integrator with a high temperature for unfolding
        integrator = LangevinIntegrator(self.temperature, 1/unit.picoseconds, self.timestep)

        # Set up the simulation
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        
        self.modeller = modeller
        self.system = system
        self.integrator = integrator
        self.simulation = simulation
        # self.positions = simulation.context.getState(getPositions=True).getPositions()
        self.topology_w_solv = modeller.topology
        self.n_nodes_w_solv = modeller.topology.getNumAtoms()
        self.n_edges_w_solv = modeller.topology.getNumBonds()
        self.topology = self.modeller_no_solvent.topology
        self.n_nodes = self.modeller_no_solvent.topology.getNumAtoms()
        self.n_edges = self.modeller_no_solvent.topology.getNumBonds()
        
        # Add reporters for data output
        simulation.reporters.append(PDBReporter(self.trajectory_file, 10000, 
                                                atomSubset = list(range(self.n_nodes))))
        simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
        
        # self.n_angles = modeller.topology.getNumAngles()
        # self.n_dihedrals = modeller.topology.getNumTorsions()
        # self.n_impropers = modeller.topology.getNumImpropers()
        # self.n_bonds = system.getNumConstraints()
        # self.n_particles = system.getNumParticles()
        # self.n_constraints = system.getNumConstraints()
        # self.n_forces = system.getNumForces()
        self.box_vectors = modeller.topology.getPeriodicBoxVectors()
        # return (self.positions , self.topology, 
            # self.n_nodes, self.n_edges, self.n_angles, self.n_dihedrals, 
            # self.n_impropers, self.n_bonds, self.n_particles, 
            # self.n_constraints, self.n_forces, self.box_vectors
    
    def run(self, epochs = 1000, steps_per_epoch=2000):
        # Run the simulation
        for i in range(epochs):
            # keep time for each step
            t0 = time.time()
            self.simulation.step(steps_per_epoch)
            # log to history 
            self.log_state(t0)
            print(f'Epoch {i+1}/{epochs}, Energy: {self.history["total_energy"][-1]:.2f}, Time: {self.history["time"][-1]:.2f} s')
            
        return self.positions
    
    def log_state(self, t0):
        dt = time.time() - t0
        state = self.simulation.context.getState(getEnergy=True, getPositions=True,)# getIntegratorParameters=True)
        E_kin = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        E_pot = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        self.history['total_energy'].append(E_kin + E_pot)
        self.history['potential_energy'].append(E_pot)
        self.history['kinetic_energy'].append(E_kin)
        # self.history['temperature'].append(state.getTemperature())
        self.history['time'].append(dt)
        
    # positions need to be updated by getting them from the simulation context
    @property
    def positions(self):
        return self.simulation.context.getState(getPositions=True).getPositions()
    
    @positions.setter
    def positions(self, positions):
        self.simulation.context.setPositions(positions)
        
    def save_pdb(self, filepath=None):
        if not filepath:
            # extract just the file name from the path
            fnam = os.path.basename(self.pdb_file).split('.')[0]
            # extract dir from pdb_file
            pdb_dir = os.path.dirname(self.pdb_file)
            n_steps = self.simulation.currentStep
            T = self.temperature.value_in_unit(unit.kelvin)
            filename = os.path.join(pdb_dir, f'fold_{fnam}-T{T}-steps{n_steps}.pdb')
        print(f'Saving pdb file to {filename}')
        PDBFile.writeFile(self.topology, self.positions[:self.n_nodes], open(filename, 'w'))            
        return filename
        
        
