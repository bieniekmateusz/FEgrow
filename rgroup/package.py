import copy
from typing import Optional

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdMolAlign import AlignMol
import py3Dmol
from MDAnalysis.analysis.distances import distance_array
import numpy as np

from .toxicity import tox_props
from .conformers import generate_conformers


def replace_atom(mol: Chem.Mol, target_idx: int, new_atom: int) -> Chem.Mol:
    edit_mol = Chem.RWMol(mol)
    for atom in edit_mol.GetAtoms():
        if atom.GetIdx() == target_idx:
            atom.SetAtomicNum(new_atom)
    return Chem.Mol(edit_mol)


def rep2D(mol, idx=True):
    numbered = copy.deepcopy(mol)
    numbered.RemoveAllConformers()
    if idx:
        for atom in numbered.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
    AllChem.Compute2DCoords(numbered)
    return numbered

def draw3D(mol, conf_id=-1):
    viewer = py3Dmol.view(width=300, height=300, viewergrid=(1,1))
    viewer.addModel(Chem.MolToMolBlock(mol, confId=conf_id), 'mol')
    viewer.setStyle({"stick":{}})
    viewer.zoomTo()
    return viewer


def draw3Dcons(mol):
    viewer = py3Dmol.view(width=300, height=300, viewergrid=(1,1))
    for i in range(mol.GetNumConformers()):
        mb = Chem.MolToMolBlock(mol, confId=i)
        viewer.addModel(mb, 'mol')
    viewer.setStyle({"stick":{}})
    viewer.zoomTo()
    return viewer



def __getAttachmentVector(R_group):
    """ for a fragment to add, search for the position of 
    the attachment point (R) and extract the atom and the connected atom 
    (currently only single bond supported)
    rgroup: fragment passed as rdkit molecule
    return: tuple (ratom, ratom_neighbour)
    """
    for atom in R_group.GetAtoms():
        if not atom.GetAtomicNum() == 0:
            continue 
        
        neighbours = atom.GetNeighbors()
        if len(neighbours) > 1:
            raise Exception("The linking R atom in the R group has two or more attachment points. "
                            "NOT IMPLEMENTED. ")
        
        return atom, neighbours[0]
    
    raise Exception('No R atom in the R group. ')


def merge_R_group(mol, R_group, replaceIndex):
    """function originally copied from
    https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py"""
    
    # the linking R atom on the R group
    rgroup_R_atom, R_atom_neighbour = __getAttachmentVector(R_group)
    print(f'Rgroup atom index {rgroup_R_atom} neighbouring {R_atom_neighbour}')
    
    # atom to be replaced in the molecule
    replace_atom = mol.GetAtomWithIdx(replaceIndex)
    assert len(replace_atom.GetNeighbors())==1, 'The atom being replaced on the molecule has more neighbour atoms than 1. Not supported.'
    replace_atom_neighbour = replace_atom.GetNeighbors()[0]
    
    # align the Rgroup
    AlignMol(R_group, mol, atomMap=(
        (R_atom_neighbour.GetIdx(),replace_atom.GetIdx()),
        (rgroup_R_atom.GetIdx(), replace_atom_neighbour.GetIdx())
                                    )
            )
    
    # merge the two molecules
    combined = Chem.CombineMols(mol, R_group)
    emol = Chem.EditableMol(combined)

    # connect
    bond_order = rgroup_R_atom.GetBonds()[0].GetBondType()
    emol.AddBond(replace_atom_neighbour.GetIdx(),
                 R_atom_neighbour.GetIdx() + mol.GetNumAtoms(),
                 order=bond_order)
    # -1 accounts for the removed linking atom on the template
    emol.RemoveAtom(rgroup_R_atom.GetIdx() + mol.GetNumAtoms())
    # remove the linking atom on the template
    emol.RemoveAtom(replace_atom.GetIdx())
    
    merged = emol.GetMol()
    Chem.SanitizeMol(merged)

    # prepare separately the template
    etemp = Chem.EditableMol(mol)
    etemp.RemoveAtom(replace_atom.GetIdx())
    template = etemp.GetMol()

    with_template = Mol(merged)
    with_template.save_template(template)

    return with_template


class Mol(rdkit.Chem.rdchem.Mol):

    def save_template(self, mol):
        self.template = mol

    def toxicity(self):
        return tox_props(self)

    def draw3D(self):
        viewer = py3Dmol.view(width=300, height=300, viewergrid=(1,1))
        viewer.addModel(Chem.MolToMolBlock(self), 'mol')
        viewer.setStyle({"stick":{}})
        viewer.zoomTo()
        return viewer

    def generate_conformers(self, num_conf: int, minimum_conf_rms: Optional[float]=None):
        cons = generate_conformers(self, num_conf, minimum_conf_rms)
        self.RemoveAllConformers()
        [self.AddConformer(con, assignId=True) for con in cons.GetConformers()]

    def draw3Dconfs(self, view=None, confids=None):
        if view is None:
            view = py3Dmol.view(width=300, height=300, viewergrid=(1,1))
            view.setStyle({"stick":{}})

        for conf in self.GetConformers():
            if confids is None:
                selected = True
            else:
                selected = True if conf.GetId() in selected else False
            mb = Chem.MolToMolBlock(self, confId=conf.GetId())
            view.addModel(mb, 'mol')
        
        view.zoomTo()
        return view

    def removeConfsClashingWithProdyProt(self, prot, min_dst_allowed=1):
        prot_coords = prot.getCoords()

        counter = 0
        for conf in list(self.GetConformers())[::-1]:
            confid = conf.GetId()

            min_dst = np.min(distance_array(conf.GetPositions(), prot_coords))

            if min_dst < min_dst_allowed:
                self.RemoveConformer(confid)
                print(f'Clash with the protein. Removing conformer id: {confid}')


