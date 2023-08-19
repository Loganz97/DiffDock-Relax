from pathlib import Path

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
def simulate_md_ligand(pdb_id:str, ligand_id:str, use_pdb_redo:bool,
                       num_steps:int|None, minimize_only:bool,
                       use_solvent:bool, decoy_smiles:str|None,
                       temperature:int, equilibration_steps:int, out_dir_root:str):
    """MD simulation of protein + ligand"""

    from MD_protein_ligand import simulate

    prepared_files = simulate.get_pdb_and_extract_ligand(pdb_id, ligand_id,
                                                         out_dir=f"{out_dir_root}/{pdb_id}_{ligand_id}",
                                                         use_pdb_redo=use_pdb_redo)

    sim_files = simulate.simulate(prepared_files["pdb"], prepared_files["sdf"],
                                  f"{out_dir_root}/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}",
                                  num_steps, minimize_only=minimize_only,
                                  use_solvent=use_solvent, decoy_smiles=decoy_smiles,
                                  temperature=temperature, equilibration_steps=equilibration_steps)

    # read in the output files
    return {out_name: (fname, open(fname, 'rb').read() if Path(fname).exists() else None)
            for out_name, fname in (prepared_files | sim_files).items()}

@stub.local_entrypoint()
def main(pdb_id:str, ligand_id:str, use_pdb_redo:bool=False,
         num_steps:int=None,
         use_solvent:bool=False, decoy_smiles:str|None=None, mutation:str|None=None,
         temperature:int=300, equilibration_steps:int=200, out_dir_root:str="out"):
    """MD simulation of protein + ligand"""

    minimize_only = True if num_steps is None else False

    outputs = simulate_md_ligand.call(pdb_id, ligand_id,
                                      use_pdb_redo, num_steps, minimize_only,
                                      use_solvent, decoy_smiles, temperature,
                                      equilibration_steps, out_dir_root)

    for _, (out_file, out_content) in outputs.items():
        if out_content:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            open(out_file, 'wb').write(out_content)

    print({k:v[0] for k, v in outputs.items()})
