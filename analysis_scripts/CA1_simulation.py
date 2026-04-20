from sys import stdout
import os
os.environ['OMP_NUM_THREADS']="64" 
import pandas as pd 
import numpy as np 


from openmm.app import *
from openmm import *
from openmm.unit import *
from pdbfixer.pdbfixer import PDBFixer, proteinResidues 

T_crys = 292
pH = 7.5 

eq_steps = 10000                 
prod_steps = 50000000             
barostatInterval = 200
pressure = 1*atmospheres
temperature = 295*kelvin
friction = 1/picosecond
timestep = 0.002*picoseconds
nonbonded_cutoff = 1*nanometer
nonbonded_method = PME
constraints = HBonds

from tqdm import tqdm

# Define total number of steps for tqdm
total_steps = 50000000

# Load PDB file
pdb = PDBFile('CA1TaCA_84695_relaxed_rank_001_alphafold2_ptm_model_2_seed_000_processed_2nm.pdb')

# Define force field and system
forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
                                 nonbondedCutoff=1*nanometer, constraints=HBonds)

# Define integrator
integrator = LangevinMiddleIntegrator(T_crys, friction, timestep)

# Create simulation
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

# Minimize energy
simulation.minimizeEnergy()
positions = simulation.context.getState(getPositions=True).getPositions()

# Define tqdm object
with tqdm(total=total_steps, desc="MD Simulation Progress") as pbar:
    # Add reporters
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
    simulation.reporters.append(StateDataReporter('20260331_CA1_timed_100ns_ProdRun_2nm.log', 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=total_steps, separator='\t'))
    simulation.reporters.append(DCDReporter('20260331_CA1_100ns_trajectory_2nm.dcd', 5000))

    # Run simulation
    for _ in range(total_steps):
        simulation.step(1)
        pbar.update(1)

# Save final positions
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('20260331_CA1_100ns_final_positions_2nm.pdb', 'w'))

