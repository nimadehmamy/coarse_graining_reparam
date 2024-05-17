from Bio.PDB import PDBParser, Superimposer, PDBIO
import py3Dmol
import os

# Example use:
# rmsd = RMSD(pdb_orig_file, pdb_test_file)
# print(rmsd.RMSD)
# view = rmsd.show_superimposed()

# Define the RMSD class
# it will be used to calculate the RMSD between the original and the predicted protein structures
# two main methods:
# - RMSD_all_atoms: calculates the RMSD between all atoms of the two structures
# - RMSD_CA_atoms: calculates the RMSD between the alpha carbon atoms of the two structures
class RMSD:
    def __init__(self,  pdb_orig_file, pdb_test_file, pdb_id=None):
        # if pdb_id is not provided, we will use the pdb_id from the pdb_orig_file
        if pdb_id is None:
            pdb_id = os.path.basename(pdb_orig_file).split('.')[0]
        self.pdb_id = pdb_id
        self.pdb_orig_file = pdb_orig_file
        self.pdb_test_file = pdb_test_file
        # get the base dir from test file 
        self.base_dir = os.path.dirname(pdb_test_file)
        # get base file name from test file
        self.base_file = os.path.basename(pdb_test_file).split('.')[0]
        
        self.get_structures()
        self.get_atoms()
        self.get_all_superimposers()
        self.get_all_RMSD()
        self.save_all_superimposed()
        
    def get_structures(self,):
        self.parser = PDBParser()
        self.structure_orig = self.parser.get_structure(f"orig-{self.pdb_id}", self.pdb_orig_file)
        self.structure_new = self.parser.get_structure(f"new-{self.pdb_id}", self.pdb_test_file)
    
    
    def get_atoms(self,):
        # Get the atoms from each structure
        self.atoms_orig = list(self.structure_orig.get_atoms())
        self.atoms_new = list(self.structure_new.get_atoms()) 
        
        # get only 'CA' atoms
        self.atoms_orig_CA = [atom for atom in self.atoms_orig if atom.get_id() == "CA"]
        self.atoms_new_CA = [atom for atom in self.atoms_new if atom.get_id() == "CA"]
        
        # get only backbone atoms
        self.atoms_orig_backbone = list(self.get_backbone_atoms(self.structure_orig))
        self.atoms_new_backbone = list(self.get_backbone_atoms(self.structure_new))
        
    def get_all_superimposers(self,):
        # for all three groups of atoms, get the superimposer
        self.superimposer_all = self.get_superimposer(self.atoms_orig, self.atoms_new)
        self.superimposer_CA = self.get_superimposer(self.atoms_orig_CA, self.atoms_new_CA)
        self.superimposer_backbone = self.get_superimposer(self.atoms_orig_backbone, self.atoms_new_backbone)
        
        
    def get_superimposer(self, atoms1, atoms2):
        # Make sure the two proteins have the same number of atoms
        assert len(atoms1) == len(atoms2), "Proteins have different numbers of atoms"
        # Create a Superimposer object
        super_imposer = Superimposer()
        # Apply the superimposition on the atom coordinates
        super_imposer.set_atoms(atoms1, atoms2)

        return super_imposer

    # Define a function to filter only backbone atoms
    def get_backbone_atoms(self, structure):
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() in ['N', 'CA', 'C']:
                            yield atom
                            
    def get_all_RMSD(self,):
        # self.RMSD_all = self.RMSD(self.atoms_orig, self.atoms_new)
        # self.RMSD_CA = self.RMSD(self.atoms_orig_CA, self.atoms_new_CA)
        # self.RMSD_backbone = self.RMSD(self.atoms_orig_backbone, self.atoms_new_backbone)
        self.RMSD_all = self.superimposer_all.rms
        self.RMSD_CA = self.superimposer_CA.rms
        self.RMSD_backbone = self.superimposer_backbone.rms
        
        self.RMSD = {'all': self.RMSD_all, 'CA': self.RMSD_CA, 'backbone': self.RMSD_backbone}
        return self.RMSD
    
    def save_all_superimposed(self,):
        # save all superimposed structures
        self.pdb_superimposed_all = self.save_superimposed(self.superimposer_all, self.structure_new, 'all')
        self.pdb_superimposed_CA = self.save_superimposed(self.superimposer_CA, self.structure_new, 'CA')
        self.pdb_superimposed_backbone = self.save_superimposed(self.superimposer_backbone, self.structure_new, 'backbone')
        
        return self.pdb_superimposed_all, self.pdb_superimposed_CA, self.pdb_superimposed_backbone
    
    def save_superimposed(self, super_imposer, structure, file_prefix=''):
        # Apply the rotation and translation to the second structure
        super_imposer.apply(structure.get_atoms())

        # Write the superimposed structure to a new PDB file
        io = PDBIO()
        io.set_structure(structure)
        pdb_superimposed_file = os.path.join(self.base_dir, f'superimposed-{self.base_file}-{file_prefix}.pdb')
        io.save(pdb_superimposed_file)
        return pdb_superimposed_file

    def show_superimposed(self,pdb_superimposed_file =None, color_orig='blue', color_new='red',):
        if pdb_superimposed_file is None:
            print("Showing all atoms superimposed structure")
            pdb_superimposed_file = self.pdb_superimposed_all
        # Read the PDB file
        with open(pdb_superimposed_file, "r") as f: 
            pdb_super = f.read()
        with open(self.pdb_orig_file, "r") as f: 
            pdb_orig = f.read()

        # Create a viewer and add the structure
        viewer = py3Dmol.view()
        viewer.addModel(pdb_super, "pdb")
        viewer.addModel(pdb_orig, "pdb")

        # Set some display options
        # viewer.setStyle({'stick': {}})
        # display the protein as a cartoon
        # viewer.setStyle({'cartoon': {'color':'spectrum'}})
        # viewer.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}})  # original structure in blue
        # viewer.setStyle({'model': 1}, {'cartoon': {'color': 'spectrum', 'opacity': 0.6}})  # superimposed structure in semi-transparent red
        viewer.setStyle({'model': 0}, {'cartoon': {'color': color_orig}})
        viewer.setStyle({'model': 1}, {'cartoon': {'color': color_new, 'opacity': 0.7}})
        
        viewer.zoomTo()

        # Show the viewer
        viewer.show()
        return viewer
