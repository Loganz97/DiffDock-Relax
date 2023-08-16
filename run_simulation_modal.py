import glob
from subprocess import run
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
def MD_test(pdb_id, ligand_id):
    """test MD"""

    from MD_protein_ligand import simulate
    print("import success")

    #
    # Example 1, a PDB file containing a ligand, minimize only!
    #
    #pdb_id, ligand_id = "6GRA", "F8Z"

    # Produces {"pdb": `out/../{pdb_id}_fixed.pdb`, "sdf": `out/../{pdb_id}_{sdf_id}.sdf`}
    prepared_files = simulate.get_pdb_and_extract_ligand(pdb_id, ligand_id,
                                                         out_dir=f"out/{pdb_id}_{ligand_id}",
                                                         use_pdb_redo=False)

    sim_files = simulate.simulate(prepared_files["pdb"], prepared_files["sdf"],
                                  f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", 10_000,
                                  use_solvent=False, decoy_smiles=None, minimize_only=True,
                                  temperature=300, equilibration_steps=200)
    print(sim_files)
    for sim_file_name, sim_file in sim_files.items():
        print(f">>> {sim_file_name}")
        print(open(sim_file, 'r', encoding='utf8').read())
        print("<<<")

@stub.local_entrypoint()
def main():
    MD_test.call("6GRA", "F8Z")
