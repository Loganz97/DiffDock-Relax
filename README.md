# Run protein-ligand MD simulation

This tool is for running an MD simulation of a protein and a ligand.

Install is a bit complex. openmm, openff, etc. require some conda.

If you have a modal labs account, you can run a simulation immediately.

### Example 1: minimize a PDB file that contains a docked ligand

### modal
```sh
modal run run_simulation_modal.py --pdb-id 4O75 --ligand-id 2RC
```

### Python
```python
from MD_protein_ligand import simulate

#
# Example 1, a PDB file containing a ligand, minimize only!
#
pdb_id, ligand_id = "4O75", "2RC"

prepared_files = simulate.get_pdb_and_extract_ligand(pdb_id, ligand_id,
                                                     out_dir=f"out/{pdb_id}_{ligand_id}",
                                                     use_pdb_redo=False)

sim_files = simulate.simulate(prepared_files["pdb"], prepared_files["sdf"],
                              f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", 10_000,
                              use_solvent=False, decoy_smiles=None, minimize_only=True,
                              temperature=300, equilibration_steps=200)
print(sim_files)
```


## Example 2: full MD trajectory for a PDB file containing a docked ligand

Note I only do 10_000 steps since this is just a demo.

### modal
```sh
modal run run_simulation_modal.py --pdb-id 4O75 --ligand-id 2RC --minimize-only False --num-steps 10_000
```

### Python
```python
pdb_id, ligand_id = "4O75", "2RC"

# Produces {"pdb": `{pdb_id}_fixed.pdb`, "sdf": `{pdb_id}_{sdf_id}.sdf`}
prepared_files = simulate.get_pdb_and_extract_ligand(pdb_id, ligand_id,
                                                     out_dir=f"out/{pdb_id}_{ligand_id}",
                                                     use_pdb_redo=False)

sim_files = simulate.simulate(prepared_files["pdb"], prepared_files["sdf"],
                              f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", 10_000,
                              use_solvent=False, decoy_smiles=None, minimize_only=False,
                              temperature=300, equilibration_steps=200)
print(sim_files)
```

## Example 3, a PDB file and a separate (docked) SDF file, minimize only

To run diffdock:
https://colab.research.google.com/drive/1CTtUGg05-2MtlWmfJhqzLTtkDDaxCDOQ
with input
pdb=4O75 smiles="Fc1cnc(nc1Nc2nc3N(C(=O)C(Oc3cc2)(C)C)COP(=O)(O)O)Nc4cc(OC)c(OC)c(OC)c4"
then
tar xvf diffdock_4O75_230605.tar
cp diffdock_4O75_230605/index0*/rank1.sdf ./diffdock_4O75_dd2RC.sdf

### Python
```python
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

