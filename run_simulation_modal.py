from pathlib import Path
from shutil import copy
import io

from modal import Image, Mount, Stub

FORCE_BUILD = False

stub = Stub()

image = (Image
         .micromamba(python_version="3.10")
         .apt_install("git", "wget")
         .micromamba_install(["openmm=8.0.0", "openmmforcefields=0.11.2", "pdbfixer=1.9",
                              "rdkit=2023.03.1", "mdtraj=1.9.9", "plotly=4.9.0",
                              "openff-toolkit=0.14.3", "python-kaleido=0.2.1", "mdanalysis=2.5.0",
                              "prody=2.4.0"],
                             channels=["omnia", "plotly", "conda-forge"])
         .pip_install("git+https://github.com/hgbrian/MD_protein_ligand", force_build=FORCE_BUILD)
        )


@stub.function(image=image, gpu="T4", timeout=60*15,
               mounts=[Mount.from_local_dir((Path(".") / "input"), remote_path="/input")])
def simulate_md_ligand(pdb_id:str, ligand_id:str, ligand_chain:str,
                       use_pdb_redo:bool, num_steps:int|None, minimize_only:bool,
                       use_solvent:bool, decoy_smiles:str|None, mutation:str|None,
                       temperature:int, equilibration_steps:int, out_dir_root:str):
    """MD simulation of protein + ligand"""

    from MD_protein_ligand import simulate
    out_dir = f"{out_dir_root}/{pdb_id}_{ligand_id}"
    out_stem = f"{out_dir}/{pdb_id}_{ligand_id}"

    if mutation is not None:
        mutate_from, mutate_resn, mutate_to, mutate_chains = mutation.split("-")
        out_stem = (f"{out_stem}_{mutate_from}_{mutate_resn}_{mutate_to}_{mutate_chains}")

    prepared_files = simulate.get_pdb_and_extract_ligand(pdb_id, ligand_id, ligand_chain,
                                                         out_dir=out_dir,
                                                         use_pdb_redo=use_pdb_redo,
                                                         mutation=(mutate_from, mutate_resn, mutate_to, mutate_chains)
                                                                  if mutation else None)

    # Read SDF file into a BytesIO object
    with open(prepared_files["sdf"], 'rb') as sdf_file:
        sdf_buffer = io.BytesIO(sdf_file.read())

    sim_files = simulate.simulate(prepared_files["pdb"], sdf_buffer, out_stem, num_steps,
                                  minimize_only=minimize_only, use_solvent=use_solvent, decoy_smiles=decoy_smiles,
                                  temperature=temperature, equilibration_steps=equilibration_steps)

    # Combine prepared_files and sim_files
    all_files = {**prepared_files, **sim_files}

    # Read in the output files
    return {out_name: (fname, open(fname, 'rb').read() if Path(fname).exists() else None)
            for out_name, fname in all_files.items()}

@stub.local_entrypoint()
def main(pdb_id:str, ligand_id:str, ligand_chain:str,
         use_pdb_redo:bool=False, num_steps:int=None,
         use_solvent:bool=True, decoy_smiles:str=None, mutation:str=None,
         temperature:int=300, equilibration_steps:int=200, out_dir_root:str="out"):
    """
    MD simulation of protein + ligand.

    :param pdb_id: String representing the PDB ID of the protein.
    :param ligand_id: String representing the ligand ID (ID in PDB file).
    :param ligand_chain: String representing the ligand chain.
        You have to explicitly specify the chain, because otherwise >1 ligands can be merged!
    :param use_pdb_redo: Boolean, whether to use PDB redo. Default is False.
    :param num_steps: Integer representing the number of steps. Default is None.
    :param use_solvent: Boolean, whether to use solvent in the simulation. Default is True.
    :param decoy_smiles: String representing the decoy SMILES notation. Default is None.
    :param mutation: String representing the mutation in the format "ALA-117-VAL-AB".
        Follows PDBFixer notation, BUT you must include the chains to mutate too. Default is None.
    :param temperature: Integer representing the temperature in the simulation. Default is 300.
    :param equilibration_steps: Integer representing the number of equilibration steps. Default is 200.
    :param out_dir_root: String representing the root directory for output. Default is "out".
    """

    minimize_only = True if not num_steps else False

    outputs = simulate_md_ligand.call(pdb_id, ligand_id, ligand_chain,
                                      use_pdb_redo, num_steps, minimize_only,
                                      use_solvent, decoy_smiles, mutation,
                                      temperature, equilibration_steps, out_dir_root)

    for (out_file, out_content) in outputs.values():
        if out_content:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'wb') as out:
                out.write(out_content)

    print("Output files:")
    for k, v in outputs.items():
        print(f"{k}: {v[0]}")
