"""
Simple protein-ligand simulation using openmm
"""

import json
import os
import re
import subprocess
import time
import io
from tempfile import NamedTemporaryFile
from pathlib import Path
from warnings import warn

import mdtraj as md
import numpy as np
import pandas as pd
import requests

import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter

from openff.toolkit.topology import Molecule
from openmm import app, unit, LangevinIntegrator, MonteCarloBarostat, Platform
from openmm.app import DCDReporter, Modeller, PDBFile, Simulation, StateDataReporter

from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms, rdShapeHelpers

# for convenience so I can use as a script or a module
try:
    from .extract_ligands import extract_ligand
except ImportError:
    from extract_ligands import extract_ligand

SMINA, SMINA_BIN = "smina", "./bin/smina"
GNINA, GNINA_BIN = "gnina", "./bin/gnina"
OBABEL, OBABEL_BIN = "obabel", "obabel"
SMINA_LINUX_URL = "https://sourceforge.net/projects/smina/files/smina.static/download"
GNINA_LINUX_URL = "https://github.com/gnina/gnina/releases/download/v1.0.3/gnina"
binaries = {SMINA: SMINA_BIN, GNINA: GNINA_BIN, OBABEL: OBABEL_BIN}

PDB_PH = 7.4
PDB_TEMPERATURE = 300 * unit.kelvin
FRICTION_COEFF = 1.0 / unit.picosecond
STEP_SIZE = 0.002 * unit.picosecond
SOLVENT_PADDING = 10.0 * unit.angstroms
BAROSTAT_PRESSURE = 1.0 * unit.atmospheres
BAROSTAT_FREQUENCY = 25
FORCEFIELD_KWARGS = {'constraints': app.HBonds, 'rigidWater': True,
                     'removeCMMotion': False, 'hydrogenMass': 4*unit.amu}
FORCEFIELD_PROTEIN = "amber/ff14SB.xml"
FORCEFIELD_SOLVENT = "amber/tip3p_standard.xml"
FORCEFIELD_SMALL_MOLECULE = "gaff-2.11"

OPENMM_DEFAULT_LIGAND_ID = "UNK"

def _download_binary_if_missing(binary_name:str):
    def _download(path, url):
        """download binary"""
        print(f"Downloading {binary_name} binary (10Mb for smina, 300Mb for gnina)")
        with requests.get(url, timeout=600, stream=True) as r:
            r.raise_for_status()
            os.makedirs(Path(path).parent, exist_ok=True)
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        os.chmod(path, 0o755)
        print(f"Downloaded {binary_name} binary\n")
        return True

    if binary_name == SMINA and not Path(SMINA_BIN).exists():
        _download(SMINA_BIN, SMINA_LINUX_URL)
    elif binary_name == GNINA and not Path(GNINA_BIN).exists():
        _download(GNINA_BIN, GNINA_LINUX_URL)

_download_binary_if_missing(SMINA)
_download_binary_if_missing(GNINA)


def get_platform():
    """Check whether we have a GPU platform and if so set the precision to mixed"""

    platform = max((Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())),
                    key=lambda p: p.getSpeed())

    if platform.getName() == 'CUDA' or platform.getName() == 'OpenCL':
        platform.setPropertyDefaultValue('Precision', 'mixed')
        print(f"Set precision for platform {platform.getName()} to mixed\n")

    return platform


def prepare_protein(in_pdb_file:str, out_pdb_file:str, minimize_pdb:bool=False,
                    mutation:tuple|None=None) -> bool:
    """
    Prepare a protein for simulation using pdbfixer and optionally minimize it using openmm.

    This function fixes common issues in PDB files and prepares them for simulation.
    It identifies missing residues, atoms, non-standard residues and heterogens, then fixes these issues.
    It also adds missing hydrogens according to the specified pH value.
    If the 'minimize_pdb' flag is set, the function additionally minimizes the energy of the
    system using a Langevin integrator.

    Parameters:
    in_pdb_file (str): Path to the input PDB file.
    out_pdb_file (str): Path to the output PDB file where the prepared protein will be saved.
    minimize_pdb (bool, optional): A flag indicating whether to minimize the PDB file using openmm.
        If it's a crystal from PDB, then you would not want to minimize it.
        If it's a docked pose, then you may want to minimize it. Defaults to False.
    mutation (tuple, optional): A mutation to apply to the protein. Defaults to None.
        Format is (from_aa, resn, to_aa, chains).
    Returns:
    bool: True if the function executes successfully, raises an exception otherwise.

    Warnings:
    This function issues a warning if DMSO is found in the PDB file, suggesting manual removal.
    """

    fixer = PDBFixer(filename=in_pdb_file)
    if mutation is not None:
        for chain in mutation[3]:
            fixer.applyMutations([f"{mutation[0]}-{mutation[1]}-{mutation[2]}"], chain)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()

    print(f"# Preparing protein:\n"
          f"- Missing residues: {fixer.missingResidues}\n"
          f"- Atoms: {fixer.missingAtoms}\n"
          f"- Terminals: {fixer.missingTerminals}\n"
          f"- Non-standard: {fixer.nonstandardResidues}\n")

    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(PDB_PH)
    fixer.removeHeterogens(keepWater=False)

    for res in fixer.topology.residues():
        if res.name == 'DMS':
            warn("DMSO found in PDB file. Maybe remove?")

    # Note in the process of adding missing residues, some weird proteins are created
    # .e.g, the PDB file could list the first N residues, but have no co-ordinates
    # Then some combination of PDBFixer and openmm place these residues but in a straight line
    with open(out_pdb_file, 'w', encoding='utf-8') as out:
        PDBFile.writeFile(fixer.topology, fixer.positions, file=out, keepIds=True)

    if minimize_pdb is True:
        system_generator = SystemGenerator(forcefields=[FORCEFIELD_PROTEIN])
        system = system_generator.create_system(fixer.topology)
        integrator = LangevinIntegrator(PDB_TEMPERATURE, FRICTION_COEFF, STEP_SIZE)
        simulation = Simulation(fixer.topology, system, integrator)
        simulation.context.setPositions(fixer.positions)
        simulation.minimizeEnergy()

        with open(f"{Path(out_pdb_file).with_suffix('')}_minimized_no_ligand.pdb", 'w',
                  encoding='utf-8') as out:
            PDBFile.writeFile(fixer.topology,
                              simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(),
                              file=out,
                              keepIds=True)

    return True

def get_pdb_and_extract_ligand(pdb_id:str,
                               ligand_id:str|None=None,
                               ligand_chain:str|None=None,
                               out_dir:str='.',
                               use_pdb_redo:bool=False,
                               minimize_pdb:bool=False,
                               mutation:tuple|None=None) -> dict:
    """
    Download a PDB file, prepare it for MD, and extract a ligand.

    This function downloads a PDB file, prepares it to a {pdb_id}_fixed.pdb file,
    then extracts one ligand and saves it as {pdb_id}_{ligand_id}.sdf and {pdb_id}_{ligand_id}.smi

    Parameters:
    pdb_id (str): The 4-letter PDB ID or path to a local PDB file.
    ligand_id (str): The 3-letter ligand ID.
    ligand_chain (str): If you want to specify the chain of the ligand. Defaults to None.
    out_dir (str): The output directory. Defaults to '.'.
    use_pdb_redo (bool): If True, use get pdb_redo PDB ({pdb_id}_pdbredo.pdb). Defaults to False.
    minimize_pdb (bool): If True, minimize the PDB file. Defaults to False.
    mutation (tuple): A mutation to apply to the protein. Format is (from_aa, resn, to_aa, chains). Defaults to None.

    Returns:
    filename_dict (dict): a dict of the files produced
    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if pdb_id.endswith(".pdb"):
        pdb_file = pdb_id
        pdb_id = Path(pdb_file).stem
    elif use_pdb_redo:
        pdb_file = str(Path(out_dir) / f"{pdb_id}_pdbredo.pdb")
        url = f"https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.pdb"
        response = requests.get(url)
        response.raise_for_status()
        with open(pdb_file, 'wb') as f:
            f.write(response.content)
    else:
        pdb_file = str(Path(out_dir) / f"{pdb_id}.pdb")
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)
        response.raise_for_status()
        with open(pdb_file, 'wb') as f:
            f.write(response.content)

    # Prepare protein
    prepared_pdb_file = str(Path(out_dir) / f"{pdb_id}_fixed.pdb")
    fixer = PDBFixer(filename=pdb_file)
    if mutation is not None:
        for chain in mutation[3]:
            fixer.applyMutations([f"{mutation[0]}-{mutation[1]}-{mutation[2]}"], chain)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    with open(prepared_pdb_file, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    if ligand_id is None:
        return {"original_pdb": pdb_file, "pdb": prepared_pdb_file}

    # Extract ligand
    out_sdf_file = str(Path(out_dir) / f"{pdb_id}_{ligand_id}.sdf")
    cmd = f"obabel -ipdb {pdb_file} -osdf -O {out_sdf_file} -l {ligand_id}"
    if ligand_chain:
        cmd += f" -c {ligand_chain}"
    subprocess.run(cmd, shell=True, check=True)

    # Generate SMILES
    mol = Chem.MolFromMolFile(out_sdf_file, sanitize=False)
    out_sdf_smiles = Chem.MolToSmiles(mol)

    return {
        "original_pdb": pdb_file,
        "pdb": prepared_pdb_file,
        "sdf": out_sdf_file,
        "smi": out_sdf_smiles
    }

def simulate(pdb_in:str, mol_in:str|io.BytesIO, output:str, num_steps:int,
             use_solvent:bool=False, decoy_smiles:str|None=None, minimize_only:bool=False,
             temperature:float=PDB_TEMPERATURE,
             equilibration_steps:int=200, reporting_interval:int|None=None,
             scoring_tool:str=GNINA) -> dict:
    """
    Run a molecular dynamics simulation using OpenMM.

    This function simulates the interactions between a protein (from a PDB file)
    and a ligand (from a MOL/SDF file or BytesIO object).
    The simulation can be performed in vacuum or in solvent.
    The simulation outputs are saved in several file formats.

    Parameters:
    pdb_in (str): Path to the input PDB file containing the protein structure.
    mol_in (str|io.BytesIO): Path to the input MOL/SDF file or BytesIO object containing the ligand structure.
    output (str): The prefix of the output file names.
    num_steps (int): The number of simulation steps to be performed.
    use_solvent (bool): If True, the simulation is performed in solvent. If False, it is performed in vacuum.
    decoy_smiles (str): The SMILES string of a decoy ligand. If not None, the decoy is used instead of the input ligand.
    minimize_only (bool): If True, only energy minimization is performed.
    temperature (float): The simulation temperature in Kelvin.
    equilibration_steps (int): The number of steps for the equilibration phase of the simulation.
    reporting_interval (int): The interval (in steps) at which the simulation data is recorded.
    scoring_tool (str): The tool to use for scoring (e.g., 'smina' or 'gnina').

    Returns:
    dict: A dictionary containing paths to all relevant output files.
    """

    os.makedirs(Path(output).parent, exist_ok=True)
    output_complex_pdb = f"{output}_complex.pdb"
    output_minimized_pdb = f"{output}_minimized.pdb"
    output_affinity_tsv = f"{output}_affinity.tsv"
    output_args_json = f"{output}_args.json"
    
    if not minimize_only:
        output_traj_dcd = f"{output}_traj.dcd"
        output_state_tsv = f"{output}_state.tsv"
        output_analysis_tsv = f"{output}_analysis.tsv"
    
    json.dump(locals(), open(output_args_json, 'w', encoding='utf-8'), indent=2)

    out_affinity = open(output_affinity_tsv, 'w', encoding='utf-8')
    out_affinity.write("time_ps\taffinity\n")

    if num_steps is None:
        num_steps = 1

    print(f"Processing {pdb_in} and {mol_in} with {num_steps} steps")

    max_frames_to_report = 100
    reporting_interval = reporting_interval or 10**(len(str(num_steps // max_frames_to_report)))

    platform = get_platform()

    print(f"# Preparing ligand:\n- {mol_in}\n")
    if isinstance(mol_in, io.BytesIO):
        ligand_rmol = Chem.MolFromMolBlock(mol_in.getvalue().decode())
        ligand_mol = Molecule(ligand_rmol)
    else:
        ligand_rmol, ligand_mol = prepare_ligand_for_MD(mol_in)
    ligand_conformer = ligand_mol.conformers[0]
    assert len(ligand_mol.conformers) == len(ligand_rmol.GetConformers()) == 1, "reference ligand should have one conformer"

    if decoy_smiles is not None:
        ligand_rmol, ligand_mol, ligand_conformer = make_decoy(ligand_rmol, decoy_smiles)
        print(f"# Using decoy:\n- {ligand_mol}\n- {ligand_conformer}\n")

    print("# Preparing system")
    system_generator = prepare_system_generator(ligand_mol, use_solvent)

    print("# Reading protein")
    protein_pdb = PDBFile(pdb_in)

    print("# Preparing complex")
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    print(f"- System has {modeller.topology.getNumAtoms()} atoms after adding protein")

    modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm())
    print(f"- System has {modeller.topology.getNumAtoms()} atoms after adding ligand")

    if use_solvent:
        modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=SOLVENT_PADDING)
        print(f"- System has {modeller.topology.getNumAtoms()} atoms after adding solvent")

    with open(output_complex_pdb, 'w', encoding='utf-8') as out:
        PDBFile.writeFile(modeller.topology, modeller.positions, out)

    affinity = get_affinity(output_complex_pdb, OPENMM_DEFAULT_LIGAND_ID, scoring_tool=scoring_tool)
    out_affinity.write(f"complex\t{affinity:.4f}\n")

    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    integrator = LangevinIntegrator(temperature, FRICTION_COEFF, STEP_SIZE)

    if use_solvent:
        system.addForce(MonteCarloBarostat(BAROSTAT_PRESSURE, temperature, BAROSTAT_FREQUENCY))

    print(f"- Using Periodic box: {system.usesPeriodicBoundaryConditions()}\n"
          f"- Default Periodic box: {system.getDefaultPeriodicBoxVectors()}\n")

    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    context = simulation.context
    context.setPositions(modeller.positions)

    print("# Minimising ...")
    simulation.minimizeEnergy()

    with open(output_minimized_pdb, 'w', encoding='utf-8') as out:
        PDBFile.writeFile(modeller.topology,
                          context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(),
                          file=out,
                          keepIds=True)

    affinity = get_affinity(output_minimized_pdb, OPENMM_DEFAULT_LIGAND_ID, scoring_tool=scoring_tool)
    out_affinity.write(f"min\t{affinity:.4f}\n")
    out_affinity.flush()

    if minimize_only:
        return {
            "complex_pdb": output_complex_pdb,
            "minimized_pdb": output_minimized_pdb,
            "affinity_tsv": output_affinity_tsv,
            "args_json": output_args_json
        }

    print("## Equilibrating ...")
    context.setVelocitiesToTemperature(temperature)
    simulation.step(equilibration_steps)

    simulation.reporters.append(DCDReporter(output_traj_dcd, reporting_interval,
                                            enforcePeriodicBox=False))
    simulation.reporters.append(StateDataReporter(output_state_tsv, reporting_interval,
                                                  step=True, potentialEnergy=True,
                                                  temperature=True))

    print(f"Starting simulation with {num_steps} steps ...")
    time_0 = time.time()
    simulation.step(num_steps)
    time_1 = time.time()
    print(f"- Simulation complete in {time_1 - time_0} seconds at {temperature}K\n")

    print("# Calculating affinities along trajectory...")
    traj_pdbs = extract_pdbs_from_dcd(output_complex_pdb, output_traj_dcd)
    traj_affinities = {time_ps: get_affinity(traj_pdb, OPENMM_DEFAULT_LIGAND_ID, scoring_tool=scoring_tool)
                       for time_ps, traj_pdb in traj_pdbs.items()}
    for time_ps, affinity in traj_affinities.items():
        out_affinity.write(f"{time_ps:.2f}\t{affinity:.4f}\n")

    print("# Running trajectory analysis...")
    _ = analyze_traj(output_traj_dcd, output_complex_pdb, output_analysis_tsv)

    (pd.read_csv(output_state_tsv, sep=',')
       .applymap(lambda x: round(x, 4) if isinstance(x, float) else x)
       .to_csv(output_state_tsv, sep='\t', index=False))

    return {
        "complex_pdb": output_complex_pdb,
        "minimized_pdb": output_minimized_pdb,
        "affinity_tsv": output_affinity_tsv,
        "args_json": output_args_json,
        "traj_dcd": output_traj_dcd,
        "state_tsv": output_state_tsv,
        "analysis_tsv": output_analysis_tsv
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenMM protein-ligand simulation")
    parser.add_argument("pdb_in", type=str, help="Input PDB file path")
    parser.add_argument("mol_in", type=str, help="Input mol file path")
    parser.add_argument("output", type=str, help="Output file name root, including path")
    parser.add_argument("num_steps", type=int, help="Number of simulation steps")
    parser.add_argument("--use_solvent", action='store_true', help="Use solvent?")
    parser.add_argument("--decoy_smiles", type=str, default=None, help="Use a decoy aligned to mol_in for simulation")
    parser.add_argument("--minimize_only", action='store_true', help="Only perform minimization")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in Kelvin")
    parser.add_argument("--equilibration_steps", type=int, default=200, help="Equilibration steps")
    parser.add_argument("--reporting_interval", type=int, default=None, help="Reporting interval")
    args = parser.parse_args()

    simulate(
        args.pdb_in,
        args.mol_in,
        args.output,
        args.num_steps,
        use_solvent=args.use_solvent,
        decoy_smiles=args.decoy_smiles,
        minimize_only=args.minimize_only,
        temperature=args.temperature,
        equilibration_steps=args.equilibration_steps,
        reporting_interval=args.reporting_interval,
    )
