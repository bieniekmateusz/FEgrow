import abc
import copy
import functools
import logging
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import List, Optional, Union
import urllib

import numpy as np
import mols2grid
import openmm
import openmm.app
import pandas
import pint_pandas
import prody as prody_package
import py3Dmol
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools

from .builder import build_molecules_with_rdkit
from .conformers import generate_conformers
from .toxicity import tox_props

# default options
pandas.set_option("display.precision", 3)

logger = logging.getLogger(__name__)


class RInterface:
    """
    This is a shared interface for a molecule and a list of molecules.

    The main purpose is to allow using the same functions on a single molecule and on a group of them.
    """

    @abc.abstractmethod
    def rep2D(self, **kwargs):
        ...

    @abc.abstractmethod
    def toxicity(self):
        pass

    @abc.abstractmethod
    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        pass

    @abc.abstractmethod
    def remove_clashing_confs(self, prot, min_dst_allowed=1):
        pass


class RMol(RInterface, rdkit.Chem.rdchem.Mol):
    """
    RMol is essentially a wrapper around RDKit Mol with
    tailored functionalities for attaching R groups, etc.

    :param rmol: when provided, energies and additional metadata is preserved.
    :type rmol: RMol
    :param template: Provide the original molecule template
        used for this RMol.
    """

    gnina_dir = None

    def __init__(self, *args, id=None, template=None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(args[0], RMol) or isinstance(args[0], rdkit.Chem.Mol):
            self.template = args[0].template if hasattr(args[0], "template") else None
            self.rgroup = args[0].rgroup if hasattr(args[0], "rgroup") else None
            self.opt_energies = (
                args[0].opt_energies if hasattr(args[0], "opt_energies") else None
            )
            self.id = args[0].id if hasattr(args[0], "id") else None
        else:
            self.template = template
            self.rgroup = None
            self.opt_energies = None
            self.id = id

    def _save_template(self, mol):
        self.template = RMol(copy.deepcopy(mol))

    def _save_opt_energies(self, energies):
        self.opt_energies = energies

    def toxicity(self):
        """
        Assessed various ADMET properties, including
         - Lipinksi rule of 5 properties,
         - the presence of unwanted substructures
         - problematic functional groups
         - synthetic accessibility

         :return: a row of a dataframe with the descriptors
         :rtype: dataframe
        """
        df = tox_props(self)
        # add an index column to the front
        df.insert(0, "ID", self.id)
        df.set_index("ID", inplace=True)

        # add a column with smiles
        df = df.assign(Smiles=[Chem.MolToSmiles(self)])

        return df

    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        """
        Generate conformers using the RDKIT's ETKDG. The generated conformers
        are embedded into the template structure. In other words,
        any atoms that are common with the template structure,
        should have the same coordinates.

        :param num_conf: fixme
        :param minimum_conf_rms: The minimum acceptable difference in the RMS in any new generated conformer.
            Conformers that are too similar are discarded.
        :type minimum_conf_rms: float
        :param flexible: A list of indices that are common with the template molecule
            that should have new coordinates.
        :type flexible: List[int]
        """
        cons = generate_conformers(self, num_conf, minimum_conf_rms, **kwargs)
        self.RemoveAllConformers()
        [self.AddConformer(con, assignId=True) for con in cons.GetConformers()]

    def optimise_in_receptor(self, *args, **kwargs):
        """
        Enumerate the conformers inside of the receptor by employing
        ANI2x, a hybrid machine learning / molecular mechanics (ML/MM) approach.
        ANI2x is neural nework potential for the ligand energetics
        but works only for the following atoms: H, C, N, O, F, S, Cl.

        Open Force Field Parsley force field is used for intermolecular interactions with the receptor.

        :param sigma_scale_factor: is used to scale the Lennard-Jones radii of the atoms.
        :param relative_permittivity: is used to scale the electrostatic interactions with the protein.
        :param water_model: can be used to set the force field for any water molecules present in the binding site.
        """
        if self.GetNumConformers() == 0:
            print("Warning: no conformers so cannot optimise_in_receptor. Ignoring.")
            return

        from .receptor import optimise_in_receptor

        opt_mol, energies = optimise_in_receptor(self, *args, **kwargs)
        # replace the conformers with the optimised ones
        self.RemoveAllConformers()
        [
            self.AddConformer(conformer, assignId=True)
            for conformer in opt_mol.GetConformers()
        ]
        # save the energies
        self._save_opt_energies(energies)

        # build a dataframe with the molecules
        conformer_ids = [c.GetId() for c in self.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [self.id] * len(energies),
                "Conformer": conformer_ids,
                "Energy": energies,
            }
        )

        return df

    def sort_conformers(self, energy_range=5):
        """
        For the given molecule and the conformer energies order the energies
         and only keep any conformers with in the energy range of the
         lowest energy conformer.

        :param energy_range: The energy range (kcal/mol),
            above the minimum, for which conformers should be kept.
        """
        if self.GetNumConformers() == 0:
            print("An rmol doesn't have any conformers. Ignoring.")
            return None
        elif self.opt_energies is None:
            raise AttributeError(
                "Please run the optimise_in_receptor in order to generate the energies first. "
            )

        from .receptor import sort_conformers

        final_mol, final_energies = sort_conformers(
            self, self.opt_energies, energy_range=energy_range
        )
        # overwrite the current confs
        self.RemoveAllConformers()
        [
            self.AddConformer(conformer, assignId=True)
            for conformer in final_mol.GetConformers()
        ]
        self._save_opt_energies(final_energies)

        # build a dataframe with the molecules
        conformer_ids = [c.GetId() for c in self.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [self.id] * len(final_energies),
                "Conformer": conformer_ids,
                "Energy": final_energies,
            }
        )

        return df

    def rep2D(self, **kwargs):
        """
        Use RDKit and get a 2D diagram.
        Uses Compute2DCoords and Draw.MolToImage function

        Works with IPython Notebook.

        :param **kwargs: are passed further to Draw.MolToImage function.
        """
        return rep2D(self, **kwargs)

    def rep3D(
        self,
        view=None,
        prody=None,
        template=False,
        confIds: Optional[List[int]] = None,
    ):
        """
        Use py3Dmol to obtain the 3D view of the molecule.

        Works with IPython Notebook.

        :param view: a view to which add the visualisation. Useful if one wants to 3D view
            multiple conformers in one view.
        :type view: py3Dmol view instance (None)
        :param prody: A prody protein around which a view 3D can be created
        :type prody: Prody instance (Default: None)
        :param template: Whether to visualise the original 3D template as well from which the molecule was made.
        :type template: bool (False)
        :param confIds: Select the conformations for display.
        :type confIds: List[int]
        """
        if prody is not None:
            view = prody_package.proteins.functions.view3D(prody)

        if view is None:
            view = py3Dmol.view(width=400, height=400, viewergrid=(1, 1))

        for conf in self.GetConformers():
            # ignore the confIds we've not asked for
            if confIds is not None and conf.GetId() not in confIds:
                continue

            mb = Chem.MolToMolBlock(self, confId=conf.GetId())
            view.addModel(mb, "lig")

            # use reverse indexing to reference the just added conformer
            # http://3dmol.csb.pitt.edu/doc/types.html#AtomSelectionSpec
            # cmap = plt.get_cmap("tab20c")
            # hex = to_hex(cmap.colors[i]).split('#')[-1]
            view.setStyle({"model": -1}, {"stick": {}})

        if template:
            mb = Chem.MolToMolBlock(self.template)
            view.addModel(mb, "template")
            # show as sticks
            view.setStyle({"model": -1}, {"stick": {"color": "0xAF10AB"}})

        # zoom to the last added model
        view.zoomTo({"model": -1})
        return view

    def remove_clashing_confs(self,
                              protein: Union[str, openmm.app.PDBFile], min_dst_allowed=1.0):
        """
        Removing conformations that class with the protein.
        Note that the original conformer should be well docked into the protein,
        ideally with some space between the area of growth and the protein,
        so that any growth on the template doesn't automatically cause
        clashes.

        :param protein: The protein against which the conformers should be tested.
        :type protein: filename or the openmm PDBFile instance or prody instance
        :param min_dst_allowed: If any atom is within this distance in a conformer, the
         conformer will be deleted.
        :type min_dst_allowed: float in Angstroms
        """
        if type(protein) is str:
            protein = openmm.app.PDBFile(protein)

        if type(protein) is openmm.app.PDBFile:
            protein_coords = protein.getPositions(asNumpy=True).in_units_of(openmm.unit.angstrom)._value
        else:
            protein_coords = protein.getCoords()

        rm_counter = 0
        for conf in list(self.GetConformers()):
            # for each atom check how far it is from the protein atoms
            min_dst = 999_999_999  # arbitrary large distance

            for point in conf.GetPositions():
                shortest = np.min(
                    np.sqrt(np.sum((point - protein_coords) ** 2, axis=1))
                )
                min_dst = min(min_dst, shortest)

                if min_dst < min_dst_allowed:
                    self.RemoveConformer(conf.GetId())
                    logger.debug(
                        f"Clash with the protein. Removing conformer id: {conf.GetId()}"
                    )
                    rm_counter += 1
                    break
        print(f"Removed {rm_counter} conformers. ")


    @staticmethod
    def set_gnina(loc):
        """
        Set the location of the binary file gnina. This could be your own compiled directory,
        or a directory where you'd like it to be downloaded.

        By default, gnina path is to the working directory (~500MB).

        :param loc: path to gnina binary file. E.g. /dir/path/gnina. Note that right now gnina should
         be a binary file with that specific filename "gnina".
        :type loc: str
        """
        # set gnina location
        path = Path(loc)
        if path.is_file():
            assert path.name == "gnina", 'Please ensure gnina binary is named "gnina"'
            RMol.gnina_dir = path.parent
        else:
            raise Exception("The path is not the binary file gnina")
        # extend this with running a binary check

    @staticmethod
    def _check_download_gnina():
        """
        Check if gnina works. Otherwise, download it.
        """
        if RMol.gnina_dir is None:
            # assume it is in the current directory
            RMol.gnina_dir = os.getcwd()

        # check if gnina works
        try:
            subprocess.run(
                ["./gnina", "--help"], capture_output=True, cwd=RMol.gnina_dir
            )
            return
        except FileNotFoundError as E:
            pass

        # gnina is not found, try downloading it
        print(f"Gnina not found or set. Download gnina (~500MB) into {RMol.gnina_dir}")
        gnina = os.path.join(RMol.gnina_dir, "gnina")
        # fixme - currently download to the working directory (Home could be more applicable).
        urllib.request.urlretrieve(
            "https://github.com/gnina/gnina/releases/download/v1.0.1/gnina",
            filename=gnina,
        )
        # make executable (chmod +x)
        mode = os.stat(gnina).st_mode
        os.chmod(gnina, mode | stat.S_IEXEC)

        # check if it works
        subprocess.run(
            ["./gnina", "--help"], capture_output=True, check=True, cwd=RMol.gnina_dir
        )

    def gnina(self, receptor_file):
        """
        Use GNINA to extract CNNaffinity, which we also recalculate to Kd (nM)

        LIMITATION: The GNINA binary does not support MAC/Windows.

        Please cite GNINA accordingly:
        McNutt, Andrew T., Paul Francoeur, Rishal Aggarwal, Tomohide Masuda, Rocco Meli, Matthew Ragoza,
        Jocelyn Sunseri, and David Ryan Koes. "GNINA 1.0: molecular docking with deep learning."
        Journal of cheminformatics 13, no. 1 (2021): 1-20.

        :param receptor_file: Path to the receptor file.
        :type receptor_file: str
        """
        self._check_download_gnina()

        # get the absolute path
        receptor = Path(receptor_file)
        if not receptor.exists():
            raise ValueError(f'Your receptor "{receptor_file}" does not seem to exist.')

        # make a temporary sdf file for gnina
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sdf")
        with Chem.SDWriter(tmp.name) as w:
            for conformer in self.GetConformers():
                w.write(self, confId=conformer.GetId())

        # run the code on the sdf
        process = subprocess.run(
            [
                os.path.join(RMol.gnina_dir, "gnina"),
                "--score_only",
                "-l",
                tmp.name,
                "-r",
                receptor,
                "--seed",
                "0",
                "--stripH",
                "False",
            ],
            capture_output=True,
            check=True,
        )

        output = process.stdout.decode("utf-8")
        CNNaffinities_str = re.findall(r"CNNaffinity: (-?\d+.\d+)", output)

        # convert to floats
        CNNaffinities = list(map(float, CNNaffinities_str))

        def ic50(x):
            return 10 ** (-x - -9)

        # generate IC50 from the CNNaffinities
        ic50s = list(map(ic50, CNNaffinities))

        # add nM units
        ic50s_nM = pandas.Series(ic50s, dtype="pint[nM]")

        # create a dataframe
        conformer_ids = [c.GetId() for c in self.GetConformers()]
        df = pandas.DataFrame(
            {
                "ID": [self.id] * len(CNNaffinities),
                "Conformer": conformer_ids,
                "CNNaffinity": CNNaffinities,
                "Kd": ic50s_nM,
            }
        )

        return df

    def to_file(self, filename: str):
        """
        Write the molecule and all conformers to file.

        Note:
            The file type is worked out from the name extension by splitting on `.`.
        """
        file_type = Path(filename).suffix.lower()

        writers = {
            ".mol": Chem.MolToMolFile,
            ".sdf": Chem.SDWriter,
            ".pdb": functools.partial(Chem.PDBWriter, flavor=1),
            ".xyz": Chem.MolToXYZFile,
        }

        func = writers.get(file_type, None)
        if func is None:
            raise RuntimeError(
                f"The file type {file_type} is not support please chose from {writers.keys()}"
            )

        if file_type in ['.pdb', '.sdf']:
            # multi-frame writers

            with writers[file_type](filename) as WRITER:
                for conformer in self.GetConformers():
                    WRITER.write(self, confId=conformer.GetId())
            return

        writers[file_type](self, filename)


    def df(self):
        """
        Generate a pandas dataframe row for this molecule with SMILES.

        :returns: pandas dataframe row.
        """
        df = pandas.DataFrame(
            {
                "ID": [self.id],
                "Smiles": [Chem.MolToSmiles(self)],
            }
        )
        # attach energies if they're present
        if self.opt_energies:
            df = df.assign(
                Energies=", ".join([str(e) for e in sorted(self.opt_energies)])
            )

        df.set_index(["ID"], inplace=True)
        return df

    def _repr_html_(self):
        # return a dataframe with the rdkit visualisation

        df = self.df()

        # add a visualisation column
        PandasTools.AddMoleculeColumnToFrame(
            df, "Smiles", "Molecule", includeFingerprints=True
        )
        return df._repr_html_()


class RList(RInterface, list):
    """
    Streamline working with RMol by presenting the same interface on the list,
    and allowing to dispatch the functions to any single member.
    """

    def rep2D(self, subImgSize=(400, 400), **kwargs):
        return Draw.MolsToGridImage(
            [mol.rep2D(rdkit_mol=True, **kwargs) for mol in self], subImgSize=subImgSize
        )

    @staticmethod
    def _append_jupyter_visualisation(df):
        if "Smiles" not in df:
            return

        # add a column with the visualisation
        Chem.PandasTools.AddMoleculeColumnToFrame(
            df, "Smiles", "Molecule", includeFingerprints=True
        )

    def toxicity(self):
        df = pandas.concat([m.toxicity() for m in self] + [pandas.DataFrame()])
        RList._append_jupyter_visualisation(df)
        return df

    def generate_conformers(
        self, num_conf: int, minimum_conf_rms: Optional[float] = [], **kwargs
    ):
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            rmol.generate_conformers(num_conf, minimum_conf_rms, **kwargs)

    def GetNumConformers(self):
        return [rmol.GetNumConformers() for rmol in self]

    def remove_clashing_confs(self, prot, min_dst_allowed=1):
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            rmol.remove_clashing_confs(prot, min_dst_allowed=min_dst_allowed)

    def optimise_in_receptor(self, *args, **kwargs):
        """
        Replace the current molecule with the optimised one. Return lists of energies.
        """
        # return pandas.concat([m.toxicity() for m in self])

        dfs = []
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            dfs.append(rmol.optimise_in_receptor(*args, **kwargs))

        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        return df

    def sort_conformers(self, energy_range=5):
        dfs = []
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            dfs.append(rmol.sort_conformers(energy_range))

        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        return df

    def gnina(self, receptor_file):
        dfs = []
        for i, rmol in enumerate(self):
            print(f"RMol index {i}")
            dfs.append(rmol.gnina(receptor_file))
	
        df = pandas.concat(dfs)
        df.set_index(["ID", "Conformer"], inplace=True)
        
        df_reset = df.reset_index()

        # Add one to the 'Conformer' column
        df_reset['Conformer'] = df_reset['Conformer'] + 1

        # Set the index back to the original state
        df_updated = df_reset.set_index(['ID', 'Conformer'])
        return df_updated

    def discard_missing(self):
        """
        Remove from this list the molecules that have no conformers
        """
        removed = []
        for rmol in self[::-1]:
            if rmol.GetNumConformers() == 0:
                rmindex = self.index(rmol)
                print(
                    f"Discarding a molecule (id {rmindex}) due to the lack of conformers. "
                )
                self.remove(rmol)
                removed.append(rmindex)
        return removed

    @property
    def dataframe(self):
        return pandas.concat([rmol.df() for rmol in self] + [pandas.DataFrame()])

    def _repr_html_(self):
        # return the dataframe with the visualisation column of the dataframe
        df = self.dataframe
        RList._append_jupyter_visualisation(df)
        return df._repr_html_()


class RGroups(pandas.DataFrame):
    """
    The default R-Group library with visualisation (mols2grid).
    """

    def __init__(self):
        data = RGroups._load_data()
        super(RGroups, self).__init__(data)

        self._fegrow_grid = mols2grid.MolGrid(self, removeHs=True, mol_col="Mol", use_coords=False, name="m2")

    @staticmethod
    def _load_data() -> pandas.DataFrame:
        """
        Load the default R-Group library

        The R-groups were largely extracted from (please cite accordingly):
        Takeuchi, Kosuke, Ryo Kunimoto, and Jürgen Bajorath. "R-group replacement database for medicinal chemistry." Future Science OA 7.8 (2021): FSO742.
        """
        molecules = []
        names = []

        builtin_rgroups = Path(__file__).parent / "data" / "rgroups" / "library.sdf"
        for rgroup in Chem.SDMolSupplier(str(builtin_rgroups), removeHs=False):
            molecules.append(rgroup)
            names.append(rgroup.GetProp("SMILES"))

            # highlight the attachment atom
            for atom in rgroup.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    setattr(rgroup, "__sssAtoms", [atom.GetIdx()])

        return {"Mol": molecules, "Name": names}

    def _ipython_display_(self):
        from IPython.display import display_html

        subset = ["img", "Name", "mols2grid-id"]
        display_html(self._fegrow_grid.display(subset=subset, substruct_highlight=True))

    def get_selected(self):
        df = self._fegrow_grid.get_selection()
        return list(df["Mol"])


class Linkers(pandas.DataFrame):
    """
    A linker library presented as a grid molecules using mols2grid library.
    """

    def __init__(self):
        # initialise self dataframe
        data = Linkers._load_data()
        super(Linkers, self).__init__(data)

        self._fegrow_grid = mols2grid.MolGrid(
            self,
            removeHs=True,
            mol_col="Mol",
            use_coords=False,
            name="m1",
            prerender=False,
        )

    @staticmethod
    def _load_data():
        # note that the linkers are pre-sorted so that:
        #  - [R1]C[R2] is next to [R2]C[R1]
        #  - according to how common they are (See the original publication) as described with SmileIndex
        builtin_rlinkers = Path(__file__).parent / "data" / "linkers" / "library.sdf"

        mols = []
        display_names = []
        smile_indices = []
        for mol in Chem.SDMolSupplier(str(builtin_rlinkers), removeHs=False):
            mols.append(mol)

            # use easier searchable SMILES, e.g. [*:1] was replaced with R1
            display_names.append(mol.GetProp("display_smiles"))

            # extract the index property from the original publication
            smile_indices.append(mol.GetIntProp("SmileIndex"))

        return {"Mol": mols, "Name": display_names, "Common": smile_indices}

    def _ipython_display_(self):
        from IPython.display import display

        subset = ["img", "Name", "mols2grid-id"]
        return display(self._fegrow_grid.display(subset=subset, substruct_highlight=True))

    def get_selected(self):
        df = self._fegrow_grid.get_selection()
        return list(df["Mol"])


def build_molecules(
    templates: Union[Chem.Mol, List[Chem.Mol]],
    r_groups: Union[Chem.Mol, List[Chem.Mol], int],
    attachment_points: Optional[List[int]] = None,
    keep_components: Optional[List[int]] = None,
):
    """

    :param templates:
    :param r_groups:
    :param attachment_points:
    :param keep_components: When the scaffold is grown from an internal atom that divides the molecules into separate
        submolecules, keep the submolecule with this atom index.
    :return:
    """

    if isinstance(r_groups, list) and len(r_groups) == 0:
        raise ValueError("Empty list received. Please pass any R-groups or R-linkers. ")

    built_mols = build_molecules_with_rdkit(
        templates, r_groups, attachment_points, keep_components
    )
    rlist = RList()
    for mol, scaffold, scaffold_no_attachement in built_mols:
        rmol = RMol(mol)

        if hasattr(scaffold, 'template') and isinstance(scaffold.template, rdkit.Chem.Mol):
            # save the original scaffold (e.g. before the linker was added)
            # this means that conformer generation will always have to regenerate the previously added R-groups/linkers
            rmol._save_template(scaffold.template)
        else:
            rmol._save_template(scaffold_no_attachement)
        rlist.append(rmol)

    return rlist


def rep2D(mol, idx=-1, rdkit_mol=False, **kwargs):
    numbered = copy.deepcopy(mol)
    numbered.RemoveAllConformers()
    if idx:
        for atom in numbered.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
    Chem.AllChem.Compute2DCoords(numbered)

    if rdkit_mol:
        return numbered
    else:
        return Draw.MolToImage(numbered, **kwargs)


def rep3D(mol):
    viewer = py3Dmol.view(width=300, height=300, viewergrid=(1, 1))
    for i in range(mol.GetNumConformers()):
        mb = Chem.MolToMolBlock(mol, confId=i)
        viewer.addModel(mb, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    return viewer
