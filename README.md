# Run protein-ligand MD simulation

This tool is for running an MD simulation of a protein and a ligand.

## How to run it

```python
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
                                                f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", 10_000,
                                                use_solvent=False, decoy_smiles=None, minimize_only=True,
                                                temperature=300, equilibration_steps=200)
print(sim_files)

#
# Example 2, a PDB file containing a ligand, full trajectory
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
# Example 3, a PDB file and a separate (docked) SDF file, minimize only
#
# To run diffdock:
# docker run --gpus all -it diffdock python diffdock_runner.py \
# 4O75 "Fc1cnc(nc1Nc2nc3N(C(=O)C(Oc3cc2)(C)C)COP(=O)(O)O)Nc4cc(OC)c(OC)c(OC)c4"
#
# gsutil cp gs://hx-bio-temp/diffdock_4O75_230605.tar .
# tar xvf diffdock_4O75_230605.tar
# cp diffdock_4O75_230605/index0*/rank1.sdf ./diffdock_4O75_dd2RC.sdf

pdb_id = "4O75"
sdf_file = "diffdock_4O75_dd2RC.sdf"
ligand_id = "dd2RC"

prepared_files = simulateComplexWithSolvent.get_pdb_and_extract_ligand(pdb_id, out_dir=f"out/{pdb_id}_{ligand_id}")
sim_files = simulateComplexWithSolvent.simulate(prepared_files["pdb"], sdf_file,
                                                f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", 10_000,
                                                use_solvent=False, decoy_smiles=None, minimize_only=True,
                                                temperature=310, equilibration_steps=200)
print(sim_files)
```

