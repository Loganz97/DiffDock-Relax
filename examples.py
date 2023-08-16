"""Test MD protein-ligand"""

import simulateComplexWithSolvent

#
# Example 1, a PDB file containing a ligand, minimize only!
#
pdb_id, ligand_id = "4O75", "2RC"

# Produces {"pdb": `{pdb_id}_fixed.pdb`, "sdf": `{pdb_id}_{sdf_id}.sdf`}
prepared_files = simulateComplexWithSolvent.get_pdb_and_extract_ligand(pdb_id, ligand_id,
                                                                       out_dir=f"out/{pdb_id}_{ligand_id}",
                                                                       use_pdb_redo=False)

sim_files = simulateComplexWithSolvent.simulate(prepared_files["pdb"], prepared_files["sdf"],
                                                f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", 1,
                                                use_solvent=False, decoy_smiles=None, minimize_only=True,
                                                temperature=300, equilibration_steps=200)
print(sim_files)

#
# Example 2, a PDB file containing a ligand, full trajectory over 100,000 steps (not enough!)
#
pdb_id, ligand_id = "4O75", "2RC"

# Produces {"pdb": `{pdb_id}_fixed.pdb`, "sdf": `{pdb_id}_{sdf_id}.sdf`}
prepared_files = simulateComplexWithSolvent.get_pdb_and_extract_ligand(pdb_id, ligand_id,
                                                                       out_dir=f"out/{pdb_id}_{ligand_id}",
                                                                       use_pdb_redo=False)

sim_files = simulateComplexWithSolvent.simulate(prepared_files["pdb"], prepared_files["sdf"],
                                                f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", 100_000,
                                                use_solvent=False, decoy_smiles=None, minimize_only=False,
                                                temperature=300, equilibration_steps=200)
print(sim_files)

#
# Example 3, a PDB file and a separate (docked) SDF file (not part of PDB file), minimize only
#

# To run diffdock: (by default it outputs to hx-bio-temp)
# docker run --gpus all -it diffdock python diffdock_runner.py 6GRA "C[C@H]1CCCC(=O)CCC/C=C/c2cc(cc(c2C(=O)O1)O)O"
# gsutil cp gs://hx-bio-temp/diffdock_6GRA_230605.tar .
# tar xvf diffdock_6GRA_230605.tar
# cp diffdock_6GRA_230605/index0*/rank1.sdf ./diffdock_6GRA_ddZEA.sdf

pdb_id = "6GRA"
sdf_file = "diffdock_6GRA_ddZEA.sdf"
ligand_id = "ddZEA"

prepared_files = simulateComplexWithSolvent.get_pdb_and_extract_ligand(pdb_id, out_dir=f"out/{pdb_id}_{ligand_id}")
sim_files = simulateComplexWithSolvent.simulate(prepared_files["pdb"], sdf_file,
                                                f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", 10_000,
                                                use_solvent=False, decoy_smiles=None, minimize_only=True,
                                                temperature=310, equilibration_steps=200)
print(sim_files)
