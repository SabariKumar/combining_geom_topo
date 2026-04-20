import mdtraj as md
import csv
import os
import numpy as np
from tqdm import tqdm

# Set directory containing trajectory and topology files
directory = '/home/oharman/20260324_WR'
trajectory_files = [f for f in os.listdir(directory) if f.endswith('_centered.dcd')]
topology_files = [f for f in os.listdir(directory) if f.endswith('_final_positions_2nm.pdb')]

# Process each trajectory and topology file pair
for traj_file in trajectory_files:
    # Extract the common identifier from the trajectory file name
    identifier = traj_file.split('_')[0]
    
    # Find the corresponding topology file based on the common identifier
    matching_top_file = [top_file for top_file in topology_files if top_file.startswith(identifier)]
    
    if not matching_top_file:
        print(f"No matching topology file found for {traj_file}")
        continue
    
    # Load the trajectory and topology files
    topology_file = os.path.join(directory, matching_top_file[0])
    trajectory_file = os.path.join(directory, traj_file)

    try:
        traj = md.load(trajectory_file, top=topology_file)
        topology = traj.topology
        print(f"Loaded trajectory {trajectory_file} with topology {topology_file} successfully.")
    except Exception as e:
        print(f"Failed to load trajectory {trajectory_file} with topology {topology_file}: {e}")
        continue

    # Define the cutoff distance for water residence calculations
    cutoff_distance = 0.75  # in nanometers (0.75 nm = 7.5 A)

    # Create a mapping of residue indices to protein atoms, using residue numbers from the PDB
    protein_residues = [residue.index for residue in topology.residues if residue.is_protein]
    protein_indices = traj.top.select('protein and (resid ' + ' '.join(map(str, protein_residues)) + ')')

    # Precompute indices for water atoms
    water_indices = traj.top.select('resname HOH')

    # Initialize the dictionary to store water presence per residue number
    residue_water_presence = {}
    for residue in topology.residues:
        if residue.is_protein:
            residue_water_presence[residue.resSeq] = []

    print(f"Selected {len(protein_indices)} protein atoms and {len(water_indices)} water atoms for {trajectory_file}.")

    # Iterate through all frames (up to 600 if available)
    num_frames = min(600, traj.n_frames)
    for frame_idx, frame in tqdm(enumerate(traj[:num_frames]), total=num_frames):
        protein_coords = frame.xyz[0, protein_indices]
        water_coords = frame.xyz[0, water_indices]

        # Compute distances between protein and water atoms in the current frame
        distances = np.linalg.norm(protein_coords[:, np.newaxis, :] - water_coords, axis=-1)

        # Identify water molecules within the cutoff distance for each residue
        frame_presence = {}
        for residue_idx in protein_residues:
            residue_distances = distances[:, residue_idx]
            residue = topology.residue(residue_idx)
            residue_presence = np.any(residue_distances <= cutoff_distance)
            frame_presence[residue.resSeq] = int(residue_presence)

        # Append the frame's presence to residue_water_presence
        for res_seq, presence in frame_presence.items():
            residue_water_presence[res_seq].append(presence)

    # Calculate normalized and mean-centered residence times per residue
    normalized_residence_times = {res_seq: sum(presence_list) / len(presence_list) for res_seq, presence_list in residue_water_presence.items()}
    mean_normalized_residence_time = np.mean(list(normalized_residence_times.values()))

    # Write the residence times grouped by residue to a CSV file
    output_filename = os.path.splitext(traj_file)[0] + '_residence_times_grouped.csv'
    try:
        with open(os.path.join(directory, output_filename), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Residue Sequence Number', 'Residue Name'] + [f'Frame_{i}' for i in range(num_frames)] + ['Normalized Residence Time', 'Mean-Centered Residence Time']
            writer.writerow(header)
            for residue in topology.residues:
                if residue.is_protein:
                    res_seq = residue.resSeq
                    presence_list = residue_water_presence[res_seq]
                    normalized_residence_time = normalized_residence_times[res_seq]
                    mean_centered_residence_time = normalized_residence_time - mean_normalized_residence_time
                    writer.writerow([res_seq, residue.name] + presence_list + [normalized_residence_time, mean_centered_residence_time])
        print(f"Successfully wrote to {output_filename}.")
    except Exception as e:
        print(f"Failed to write to CSV {output_filename}: {e}")

    print(f"Completed processing for {trajectory_file}.")