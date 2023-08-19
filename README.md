# Run protein-ligand MD simulation

This tool is for running an MD simulation of a protein and a ligand.
It is based off of [simple-simulate-complex](https://github.com/tdudgeon/simple-simulate-complex)
which did most of the hard work of
getting expert input on figuring out reasonable parameters.

See the
[simple-simulate-complex README](https://github.com/tdudgeon/simple-simulate-complex/blob/master/README.md)
for details.

The libraries required make installion a bit complex.
Generally, openmm, openff, etc. seem to work best with conda.

If you have a modal labs account, you can run a simulation immediately with one line of code.

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
                              f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", None,
                              use_solvent=False, decoy_smiles=None, minimize_only=True,
                              temperature=310, equilibration_steps=200)
print(sim_files)
```


## Example 2: full MD trajectory for a PDB file containing a docked ligand

Note I only do 10_000 steps since this is just a demo.

### modal
```sh
modal run run_simulation_modal.py --pdb-id 4O75 --ligand-id 2RC --num-steps 10_000
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
                              temperature=310, equilibration_steps=200)
print(sim_files)
```

## Example 3, a PDB file and a separate (docked) SDF file, minimize only

First run the [DiffDock colab](https://colab.research.google.com/drive/1CTtUGg05-2MtlWmfJhqzLTtkDDaxCDOQ)
with input
- pdb_id = "4O75"
- smiles = "Fc1cnc(nc1Nc2nc3N(C(=O)C(Oc3cc2)(C)C)COP(=O)(O)O)Nc4cc(OC)c(OC)c(OC)c4"

```sh
tar xvf diffdock_4O75_230605.tar
cp diffdock_4O75_230605/index0*/rank1.sdf ./input/diffdock_4O75_dd2RC.sdf
```

### modal
```sh
modal run run_simulation_modal.py --pdb-id 4O75 --ligand-id ./input/diffdock_4O75_dd2RC.sdf
```

### Python
```python
pdb_id = "4O75"
sdf_file = "input/diffdock_4O75_dd2RC.sdf"
ligand_id = "dd2RC"

prepared_files = simulate.get_pdb_and_extract_ligand(pdb_id, out_dir=f"out/{pdb_id}_{ligand_id}")
sim_files = simulate.simulate(prepared_files["pdb"], sdf_file,
                              f"out/{pdb_id}_{ligand_id}/{pdb_id}_{ligand_id}", None,
                              use_solvent=False, decoy_smiles=None, minimize_only=True,
                              temperature=310, equilibration_steps=200)
print(sim_files)
```

## Example 4: mutate amino acid and evaluate

Introduce mutations from forward genetics into protein
and test affinity after relaxing with MD
(since mutations alter the protein structure)

### Python

```python
from pathlib import Path
from MD_protein_ligand.mutate_pdb import mutate_pdb
from MD_protein_ligand.simulate import get_pdb_and_extract_ligand, simulate

# mutations that may confer resistance
mutations = [
    ("L", 505, "H"),
]

# Protein / mutation params
# generally mutate ALL chains in multimer just in case
pdb_id, ligand_id = "3OG7", "032"
chains_to_mutate = "AB"
out_dir = f"out/{pdb_id}_{ligand_id}"
use_pdb_redo = False

# Simulation params
# use_solvent makes it take 10x longer or more and may run out of memory
# decoy_smiles is not necessary here, since we are comparing mutations, not ligands
# If minimize_only=True, then the number of steps does not matter
# temperature set to ~37C
# equilibriation_steps is 200 by default
SIM_PARAMS = dict(use_solvent=False, decoy_smiles=None, minimize_only=True, num_steps=None,
                  temperature=310, equilibration_steps=200)

prepared_files = get_pdb_and_extract_ligand(pdb_id, ligand_id, out_dir=out_dir,
                                            use_pdb_redo=use_pdb_redo)
print(f"{prepared_files=}")

#
# Minimize and get affinity from gnina for original PDB
#
sim_files = simulate(prepared_files["pdb"], prepared_files["sdf"],
                     f"{out_dir}/{pdb_id}_{ligand_id}",
                     **SIM_PARAMS)
print(f"{sim_files=}")

print("Affinity",
      [l for l in open(sim_files["smina_affinity_tsv"]) if l.startswith("min")][0])

#
# Now do the same for each mutation
#
for from_aa, res_num, to_aa in mutations:
    mutated_pdb_file = mutate_pdb(prepared_files["original_pdb"], chains_to_mutate, res_num,
                                  to_aa, check_original_aa=from_aa)

    mutated_prepared_files = get_pdb_and_extract_ligand(mutated_pdb_file, out_dir=out_dir,
                                                        use_pdb_redo=use_pdb_redo)

    sim_files = simulate(mutated_prepared_files["pdb"], prepared_files["sdf"],
                         f"{out_dir}/{Path(mutated_pdb_file).stem}_{ligand_id}",
                         **SIM_PARAMS)
    print("Affinity", from_aa, res_num, to_aa,
          [l for l in open(sim_files["smina_affinity_tsv"]) if l.startswith("min")][0])

```