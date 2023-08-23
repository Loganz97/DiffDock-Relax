from pathlib import Path
from shutil import copy

from modal import Image, Mount, Stub

stub = Stub()

image = (Image
         .micromamba(python_version="3.10")
         .apt_install("git", "wget")
         .micromamba_install(["openmm=8.0.0", "openmmforcefields=0.11.2", "pdbfixer=1.9",
                              "rdkit=2023.03.1", "mdtraj=1.9.9", "plotly=4.9.0",
                              "openff-toolkit=0.14.3", "python-kaleido=0.2.1", "mdanalysis=2.5.0",
                              "prody=2.4.0"],
                             channels=["omnia", "plotly", "conda-forge"])
         .pip_install("git+https://github.com/hgbrian/MD_protein_ligand")
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

    prepared_files = simulate.get_pdb_and_extract_ligand(pdb_id, ligand_id, ligand_chain,
                                                         out_dir=out_dir,
                                                         use_pdb_redo=use_pdb_redo)

    #
    # Mutate a residue and relax again.
    # Mutate prepared_files["pdb"] to ensure consistency
    # e.g., LEU-117-VAL-AB, following PDBFixer format (but adding chains)
    #
    if mutation is not None:
        mutate_from, mutate_resn, mutate_to, mutate_chains = mutation.split("-")
        out_stem = (f"{out_dir}/{Path(prepared_files['pdb']).stem}_{mutate_from}{mutate_resn}{mutate_to}_{mutate_chains}")
        copy(prepared_files["pdb"], f"{out_stem}.pdb")
        mutated_pdb = simulate.get_pdb_and_extract_ligand(f"{out_stem}.pdb", out_dir=out_dir,
                                                          use_pdb_redo=use_pdb_redo,
                                                          mutation=(mutate_from, mutate_resn, mutate_to, mutate_chains))
        prepared_files["pdb"] = mutated_pdb["pdb"]

    sim_files = simulate.simulate(prepared_files["pdb"], prepared_files["sdf"], out_stem, num_steps, 
                                  minimize_only=minimize_only, use_solvent=use_solvent, decoy_smiles=decoy_smiles,
                                  temperature=temperature, equilibration_steps=equilibration_steps)

    # read in the output files
    return {out_name: (fname, open(fname, 'rb').read() if Path(fname).exists() else None)
            for out_name, fname in (prepared_files | sim_files).items()}

@stub.local_entrypoint()
def main(pdb_id:str, ligand_id:str, ligand_chain:str,
         use_pdb_redo:bool=False, num_steps:int=None,
         use_solvent:bool=False, decoy_smiles:str=None, mutation:str=None,
         temperature:int=300, equilibration_steps:int=200, out_dir_root:str="out"):
    """MD simulation of protein + ligand
    mutation is a string like "ALA-117-VAL-AB"
    """

    minimize_only = True if not num_steps else False

    # original
    outputs = simulate_md_ligand.call(pdb_id, ligand_id, ligand_chain,
                                      use_pdb_redo, num_steps, minimize_only,
                                      use_solvent, decoy_smiles, mutation,
                                      temperature, equilibration_steps, out_dir_root)

    for (out_file, out_content) in outputs.values():
        if out_content:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            open(out_file, 'wb').write(out_content)

    print({k:v[0] for k, v in outputs.items()})
