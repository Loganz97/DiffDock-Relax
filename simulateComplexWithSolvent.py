"""
Simple protein-ligand simluation using openmm
"""

import argparse
import numpy as np
import os
import subprocess
import time

from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SystemGenerator

from openmm import app, unit, Platform, LangevinIntegrator, MonteCarloBarostat
from openmm.app import PDBFile, Simulation, Modeller, StateDataReporter, DCDReporter

import mdtraj as md
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms, rdShapeHelpers

FRICTION_COEFF = 1.0 / unit.picosecond
STEP_SIZE = 0.002 * unit.picoseconds
SOLVENT_PADDING = 10.0 * unit.angstroms
BAROSTAT_PRESSURE = 1.0 * unit.atmospheres
BAROSTAT_FREQUENCY = 25
FORCEFIELD_KWARGS = {'constraints': app.HBonds, 'rigidWater': True, 
                     'removeCMMotion': False, 'hydrogenMass': 4*unit.amu}


def get_platform():
    """Check whether we have a GPU platform and if so set the precision to mixed"""

    platform = max((Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())), 
                    key=lambda p: p.getSpeed())

    if platform.getName() == 'CUDA' or platform.getName() == 'OpenCL':
        platform.setPropertyDefaultValue('Precision', 'mixed')
        print(f"Set precision for platform {platform.getName()} to mixed")

    return platform


def get_pdb_and_extract_ligand(pdb_id, sdf_id):
    """Download PDB file, extract one ligand and save as SDF file"""

    if not os.path.exists(f"{pdb_id}.pdb"):
        subprocess.run(f"wget https://files.rcsb.org/download/{pdb_id}.pdb", check=True, shell=True)


    if not os.path.exists(f"{pdb_id}_{sdf_id}.sdf") or not os.path.exists(f"{pdb_id}_h.pdb"):
        is_hetatm = False
        is_ligand = False

        with (open(f"/tmp/{sdf_id}.pdb", 'w', encoding='utf-8') as out_sdf,
              open(f"/tmp/{pdb_id}.pdb", 'w', encoding='utf-8') as out_pdb):
            for line in open(f"{pdb_id}.pdb", encoding='utf-8'):
                if line.startswith("HETATM"):
                    is_hetatm = True
                    is_ligand = False
                    if line[17:20] == sdf_id:
                        out_sdf.write(line)
                        is_ligand = True
                elif is_hetatm is True and line.startswith("ANISOU"):
                    is_hetatm = False
                    if is_ligand is True:
                        out_sdf.write(line)
                        is_ligand = False
                else:
                    out_pdb.write(line)
                    is_hetatm = False

        subprocess.run(f"pdbfixer /tmp/{pdb_id}.pdb --keep-heterogens=none --output={pdb_id}_h.pdb", check=True, shell=True)

        # -d deletes Hydrogens, since the simulation code will add hydrogens?
        subprocess.run(f"obabel /tmp/{sdf_id}.pdb -O {pdb_id}_{sdf_id}.sdf -d", check=True, shell=True)

    return True


def make_decoy(reference_mol, decoy_smiles, num_conformers = 100):
    """given a molecule and a decoy smiles,
      generate a decoy molecule and its closest conformer in terms of shape"""

    assert len(reference_mol.GetConformers()) == 1, "reference ligand should be fixed"

    # Convert SMILES to 3D structure
    decoy_mol = Chem.MolFromSmiles(decoy_smiles)
    decoy_mol = Chem.AddHs(decoy_mol)
    #AllChem.EmbedMolecule(decoy_mol) # unnecessary?

    # Generate conformers
    AllChem.EmbedMultipleConfs(decoy_mol, numConfs=num_conformers)

    # Align each conformer to the original ligand
    centroid_mol = rdMolTransforms.ComputeCentroid(reference_mol.GetConformer())

    min_shape_dist = None
    for n, conformer in enumerate(decoy_mol.GetConformers()):
        centroid_decoy = rdMolTransforms.ComputeCentroid(conformer)
        translation = centroid_mol - centroid_decoy
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = translation

        rdMolTransforms.TransformConformer(conformer, transformation_matrix)
        shape_dist = rdShapeHelpers.ShapeTanimotoDist(reference_mol, decoy_mol, confId1=0, confId2=n)
        if min_shape_dist is None or shape_dist < min_shape_dist:
            best_conformer = n
            min_shape_dist = shape_dist

    print(best_conformer, min_shape_dist)
    return decoy_mol, best_conformer


def prepare_ligand(mol_in):
    """Read the sdf/mol file into RDKit, add Hs and create an openforcefield Molecule object"""

    print("Reading ligand")
    rdkitmol = Chem.MolFromMolFile(mol_in)

    print("Adding hydrogens")
    rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)

    # Ensure the chiral centers are all defined
    Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)

    return rdkitmolh, Molecule(rdkitmolh)


def prepare_system_generator(ligand_mol=None, use_solvent=False):
    """Prepare system generator"""

    if use_solvent:
        system_generator = SystemGenerator(
            forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
            small_molecule_forcefield='gaff-2.11',
            molecules=[ligand_mol],
            forcefield_kwargs=FORCEFIELD_KWARGS)
    else:
        system_generator = SystemGenerator(
            forcefields=['amber/ff14SB.xml'],
            small_molecule_forcefield='gaff-2.11',
            forcefield_kwargs=FORCEFIELD_KWARGS)

    return system_generator


def analyze_traj(traj_in, topol_in, output_analysis):
    """Run analysis of RMSD for backbone and ligand using mdtraj"""

    t = md.load(traj_in, top=topol_in)
    print(f"Topology: {t.topology} with n_frames={t.n_frames}")

    lig_atoms = t.topology.select("chainid 1")
    print(f"{len(lig_atoms)} ligand atoms")

    rmsds_lig = md.rmsd(t, t, frame=0, atom_indices=lig_atoms, parallel=True, precentered=False)
    print(f"rmsds_lig {rmsds_lig}")

    bb_atoms = t.topology.select("chainid 0 and backbone")
    print(f"{len(bb_atoms)} backbone atoms")

    rmsds_bck = md.rmsd(t, t, frame=0, atom_indices=bb_atoms, parallel=True, precentered=False)

    df_traj = (pd.DataFrame([t.time, rmsds_bck, rmsds_lig]).T
                 .applymap(lambda x: round(x, 8))
                 .rename(columns={0:'time', 1:'rmsd_bck', 2:'rmsd_lig'}))

    df_traj.to_csv(output_analysis, sep='\t', index=False)

    return df_traj


def simulate(pdb_in, mol_in, output, num_steps,
             use_solvent, temperature, equilibration_steps, reporting_interval):
    """Run simulation"""

    output_complex = f"{output}_complex.pdb"
    output_traj_pdb = f"{output}_traj.pdb"
    output_traj_dcd = f"{output}_traj.dcd"
    output_min = f"{output}_minimised.pdb"
    output_state = f"{output}_state.tsv"
    output_analysis = f"{output}_analysis.tsv"

    print(f"Processing {pdb_in} and {mol_in} with {num_steps} steps generating outputs: "
          f"{output_complex}, {output_min}, {output_traj_pdb}, {output_traj_dcd}")

    platform = get_platform()

    print("Preparing ligand")
    rdkitmolh, ligand_mol = prepare_ligand(mol_in)

    # Initialize a SystemGenerator using the GAFF for the ligand and tip3p for the water.
    # Chat-GPT: To use a larger time step, artificially increase the mass of the hydrogens.
    print("Preparing system")
    system_generator = prepare_system_generator(ligand_mol, use_solvent)

    # Use Modeller to combine the protein and ligand into a complex
    print("Reading protein")
    protein_pdb = PDBFile(pdb_in)

    # real ligand
    decoy_smiles = "Cc1ccccc1NC(=O)Nc2c(n(c(=S)s2)C)C"
    # zearalenone
    decoy_smiles = "C[C@H]1CCCC(=O)CCC/C=C/C2=C(C(=CC(=C2)O)O)C(=O)O1"
    # staurosporine
    decoy_smiles = "C[C@@]12[C@@H]([C@@H](C[C@@H](O1)N3C4=CC=CC=C4C5=C6C(=C7C8=CC=CC=C8N2C7=C53)CNC6=O)NC)OC"
    decoy_mol, decoy_conformer = make_decoy(rdkitmolh, decoy_smiles)
    print("decoy", decoy_mol, decoy_conformer)

    print("Preparing complex")
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    print(f"System has {modeller.topology.getNumAtoms()} atoms")

    # This next bit is black magic.
    # Modeller needs topology and positions. Lots of trial and error found that this is what works to get
    # these from an openforcefield Molecule object that was created from a RDKit molecule.
    # The topology part is described in the openforcefield API but the positions part grabs the first
    # (and only) conformer and passes it to Modeller. It works. Don't ask why!
    # modeller.topology.setPeriodicBoxVectors([Vec3(x=8.461, y=0.0, z=0.0), 
    # Vec3(x=0.0, y=8.461, z=0.0), Vec3(x=0.0, y=0.0, z=8.461)])
    modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm())
    print(f"System has {modeller.topology.getNumAtoms()} atoms")

    if use_solvent:
        # We use the 'padding' option to define the periodic box. The PDB file does not contain any
        # unit cell information so we just create a box that has a 10A padding around the complex.
        print("Adding solvent...")
        modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=SOLVENT_PADDING)
        print(f"System has {modeller.topology.getNumAtoms()} atoms")

    # Output the complex with topology
    with open(output_complex, 'w', encoding='utf-8') as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

    # Create the system using the SystemGenerator
    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    integrator = LangevinIntegrator(temperature, FRICTION_COEFF, STEP_SIZE)

    # This line is present in the WithSolvent.py version of the script but unclear why
    if use_solvent:
        system.addForce(MonteCarloBarostat(BAROSTAT_PRESSURE, temperature, BAROSTAT_FREQUENCY))

    print(f"Uses Periodic box: {system.usesPeriodicBoundaryConditions()}\n"
          f"Default Periodic box: {system.getDefaultPeriodicBoxVectors()}")

    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    context = simulation.context
    context.setPositions(modeller.positions)

    print("Minimising ...")
    simulation.minimizeEnergy()

    # Write out the minimised PDB. 'enforcePeriodicBox=False' is important otherwise the different
    # components can end up in different periodic boxes resulting in really strange looking output.
    with open(output_min, 'w', encoding='utf-8') as outfile:
        PDBFile.writeFile(modeller.topology,
                          context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(),
                          file=outfile,
                          keepIds=True)

    print("Equilibrating ...")
    context.setVelocitiesToTemperature(temperature)
    simulation.step(equilibration_steps)

    # Run the simulation.
    # The enforcePeriodicBox arg to the reporters is important.
    # It's a bit counter-intuitive that the value needs to be False, but this is needed to ensure
    # that all parts of the simulation end up in the same periodic box when being output.
    # simulation.reporters.append(PDBReporter(output_traj_pdb, reporting_interval, enforcePeriodicBox=False))
    simulation.reporters.append(DCDReporter(output_traj_dcd, reporting_interval,
                                            enforcePeriodicBox=False))
    simulation.reporters.append(StateDataReporter(output_state, reporting_interval,
                                                  step=True, potentialEnergy=True, temperature=True))

    print(f"Starting simulation with {num_steps} steps ...")
    time_0 = time.time()
    simulation.step(num_steps)
    time_1 = time.time()
    print(f"Simulation complete in {time_1 - time_0} seconds at {temperature} K")

    print("Running analysis...")
    df_traj = analyze_traj(output_traj_dcd, output_complex, output_analysis)

    # fix the state data file from csv to tsv
    (pd.read_csv(output_state, sep=',')
       .applymap(lambda x: round(x, 4) if isinstance(x, float) else x)
       .to_csv(output_state, sep='\t', index=False))

    return df_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenMM Simulation")
    parser.add_argument("pdb_in", type=str, help="Input PDB file")
    parser.add_argument("mol_in", type=str, help="Input mol file")
    parser.add_argument("output", type=str, help="Output file name root")
    parser.add_argument("num_steps", type=int, help="Number of simulation steps")
    parser.add_argument("--use_solvent", action='store_true', help="Use solvent?")
    parser.add_argument("--temperature", type=float, default=300, help="Temperature in Kelvin")
    parser.add_argument("--equilibration_steps", type=int, default=200, help="Equilibration steps")
    parser.add_argument("--reporting_interval", type=int, default=1000, help="Reporting interval")
    args = parser.parse_args()

    simulate(
        args.pdb_in,
        args.mol_in,
        args.output,
        args.num_steps,
        args.use_solvent,
        args.temperature,
        args.equilibration_steps,
        args.reporting_interval,
    )
