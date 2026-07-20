from lammps import lammps
from mpi4py import MPI
import os
import math
import time
from collections import defaultdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

n_cells_per_dir = 6
n_crosslink_nodes = n_cells_per_dir ** 3
box_volume = 36**3
L_c = 14 #adjustable

run = f'{L_c}-{str(n_cells_per_dir)}'
amplitude = '0.00'  #adjustable
np_radius = 1.8
frequency = 0.002

#these are used for cutoffs
wca = 1.12246
np_polymer_contact = 0.5 + np_radius
np_np_contact = 2 * np_radius

trials = 5
equilibration_steps = 100_000
run_steps = 5_000_000

np_mass = (np_radius * 2) ** 3


def min_image(dx, L):
    if dx > 0.5 * L:
        dx -= L
    elif dx < -0.5 * L:
        dx += L
    return dx

def bond_length(pos1, pos2, Lx, Ly, Lz):
    dx = min_image(pos2[0] - pos1[0], Lx)
    dy = min_image(pos2[1] - pos1[1], Ly)
    dz = min_image(pos2[2] - pos1[2], Lz)
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def measure_contour_lengths(data_file, n_crosslink_nodes, output_file):
    atoms = {}
    bonds = []

    Lx = Ly = Lz = None

    section = None

    with open(data_file, "r") as f:
        for line in f:
            stripped = line.strip()

            if stripped == "":
                continue

            if "xlo xhi" in stripped:
                parts = stripped.split()
                xlo = float(parts[0])
                xhi = float(parts[1])
                Lx = xhi - xlo
                continue

            if "ylo yhi" in stripped:
                parts = stripped.split()
                ylo = float(parts[0])
                yhi = float(parts[1])
                Ly = yhi - ylo
                continue

            if "zlo zhi" in stripped:
                parts = stripped.split()
                zlo = float(parts[0])
                zhi = float(parts[1])
                Lz = zhi - zlo
                continue

            if stripped.startswith("Atoms"):
                section = "Atoms"
                continue

            if stripped.startswith("Bonds"):
                section = "Bonds"
                continue

            if stripped.startswith("Velocities") or stripped.startswith("Masses"):
                section = None
                continue

            if section == "Atoms":
                parts = stripped.split()
                if len(parts) >= 7 and parts[0].isdigit():
                    atom_id = int(parts[0])
                    x = float(parts[4])
                    y = float(parts[5])
                    z = float(parts[6])
                    atoms[atom_id] = (x, y, z)

            elif section == "Bonds":
                parts = stripped.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    a = int(parts[2])
                    b = int(parts[3])
                    bonds.append((a, b))


    adjacency = defaultdict(list)

    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)

    crosslinks = set(range(1, n_crosslink_nodes + 1))

    strand_lengths = []
    visited_edges = set()

    for start in crosslinks:
        for neighbor in adjacency[start]:

            edge_key = tuple(sorted((start, neighbor)))
            if edge_key in visited_edges:
                continue

            length = 0.0
            previous = start
            current = neighbor

            visited_edges.add(edge_key)
            length += bond_length(atoms[start], atoms[neighbor], Lx, Ly, Lz)

            while current not in crosslinks:
                next_candidates = [x for x in adjacency[current] if x != previous]

                nxt = next_candidates[0]

                edge_key = tuple(sorted((current, nxt)))
                visited_edges.add(edge_key)

                length += bond_length(atoms[current], atoms[nxt], Lx, Ly, Lz)

                previous = current
                current = nxt

            end = current

            strand_lengths.append((start, end, length))

    lengths_only = [x[2] for x in strand_lengths]

    mean_Lc = sum(lengths_only) / len(lengths_only)
    min_Lc = min(lengths_only)
    max_Lc = max(lengths_only)

    variance = sum((x - mean_Lc) ** 2 for x in lengths_only) / len(lengths_only)
    std_Lc = math.sqrt(variance)

    with open(output_file, "w") as f:
        f.write("# Contour length statistics\n")
        f.write(f"# Number of strands: {len(strand_lengths)}\n")
        f.write(f"# Mean contour length: {mean_Lc:.8f}\n")
        f.write(f"# Std contour length: {std_Lc:.8f}\n")
        f.write(f"# Min contour length: {min_Lc:.8f}\n")
        f.write(f"# Max contour length: {max_Lc:.8f}\n")
        f.write("#\n")

    return mean_Lc, std_Lc, min_Lc, max_Lc, len(strand_lengths)


def find_strand_endpoints(data_file, n_crosslink_nodes):
    bonds = []
    section = None

    with open(data_file, "r") as f:
        for line in f:
            stripped = line.strip()

            if stripped == "":
                continue

            if stripped.startswith("Bonds"):
                section = "Bonds"
                continue

            if stripped.startswith("Velocities") or stripped.startswith("Masses"):
                section = None
                continue

            if section == "Bonds":
                parts = stripped.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    a = int(parts[2])
                    b = int(parts[3])
                    bonds.append((a, b))

    adjacency = defaultdict(list)

    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)

    crosslinks = set(range(1, n_crosslink_nodes + 1))
    strand_endpoints = []
    visited_edges = set()

    for start in crosslinks:
        for neighbor in adjacency[start]:
            edge_key = tuple(sorted((start, neighbor)))
            if edge_key in visited_edges:
                continue

            previous = start
            current = neighbor
            visited_edges.add(edge_key)

            while current not in crosslinks:
                next_candidates = [x for x in adjacency[current] if x != previous]

                nxt = next_candidates[0]
                edge_key = tuple(sorted((current, nxt)))
                visited_edges.add(edge_key)

                previous = current
                current = nxt

            strand_endpoints.append((start, current))

    return strand_endpoints


def measure_mesh_size_distribution(mesh_trajectory_file, strand_endpoints, distribution_output_file):
    mesh_samples = []

    with open(mesh_trajectory_file, "r") as f:
        while True:
            line = f.readline()
            if line == "":
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(f.readline().strip())

            number_of_atoms_header = f.readline().strip()
            if not number_of_atoms_header.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(
                    f"Expected 'ITEM: NUMBER OF ATOMS' after timestep {step}, "
                    f"but found: {number_of_atoms_header!r}"
                )

            n_atoms = int(f.readline().strip())

            box_header = f.readline().strip()

            bounds_1 = [float(x) for x in f.readline().split()]
            bounds_2 = [float(x) for x in f.readline().split()]
            bounds_3 = [float(x) for x in f.readline().split()]

            if len(bounds_1) == 3:
                xlo_bound, xhi_bound, xy = bounds_1
                ylo_bound, yhi_bound, xz = bounds_2
                zlo, zhi, yz = bounds_3

                xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
                xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
                ylo = ylo_bound - min(0.0, yz)
                yhi = yhi_bound - max(0.0, yz)
            else:
                xlo, xhi = bounds_1
                ylo, yhi = bounds_2
                zlo, zhi = bounds_3
                xy = 0.0
                xz = 0.0
                yz = 0.0

            Lx = xhi - xlo
            Ly = yhi - ylo
            Lz = zhi - zlo

            atom_header = f.readline().strip()
            scaled_positions = {}

            for _ in range(n_atoms):
                parts = f.readline().split()
                atom_id = int(parts[0])
                scaled_positions[atom_id] = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                )

            for start, end in strand_endpoints:
                s1 = scaled_positions[start]
                s2 = scaled_positions[end]

                dsx = s2[0] - s1[0]
                dsy = s2[1] - s1[1]
                dsz = s2[2] - s1[2]

                dsx -= round(dsx)
                dsy -= round(dsy)
                dsz -= round(dsz)

                dx = Lx * dsx + xy * dsy + xz * dsz
                dy = Ly * dsy + yz * dsz
                dz = Lz * dsz

                mesh_size = math.sqrt(dx * dx + dy * dy + dz * dz)
                mesh_samples.append((step, start, end, mesh_size))


    mesh_sizes_only = [sample[3] for sample in mesh_samples]
    mean_mesh_size = sum(mesh_sizes_only) / len(mesh_sizes_only)

    variance = sum(
        (mesh_size - mean_mesh_size) ** 2
        for mesh_size in mesh_sizes_only
    ) / len(mesh_sizes_only)

    std_mesh_size = math.sqrt(variance)

    with open(distribution_output_file, "w") as f:
        f.write("# Mesh size distribution sampled during the production run\n")
        f.write("# step crosslink_1 crosslink_2 mesh_size\n")

        for step, start, end, mesh_size in mesh_samples:
            f.write(f"{step} {start} {end} {mesh_size:.8f}\n")

    return mean_mesh_size, std_mesh_size, len(mesh_samples)


overall_start = MPI.Wtime()

for trial in range(trials):
    comm.Barrier()
    trial_start = MPI.Wtime()

    if rank == 0:
        print(f"RUNNING TRIAL {trial + 1} using {n_ranks} MPI processes")

    log_dir = f"lammps/{run}/log{amplitude}_trial{trial+1}.lammps"
    traj_dir = f"lammps/{run}/trajectory{amplitude}_trial{trial+1}.lammpstrj"
    temp_dir = f"lammps/{run}/temperature{amplitude}_trial{trial+1}.txt"
    equilibrated_data_dir = (
        f"lammps/{run}/equilibrated{amplitude}_trial{trial + 1}.data"
    )
    contour_dir = (
        f"lammps/{run}/contour_lengths{amplitude}_trial{trial + 1}.txt"
    )
    mesh_traj_dir = (
        f"lammps/{run}/mesh_trajectory{amplitude}_trial{trial + 1}.lammpstrj"
    )
    mesh_distribution_dir = (
        f"lammps/{run}/mesh_distribution{amplitude}_trial{trial + 1}.txt"
    )

    polymer_seed = 12345 + 1000 * trial
    np_seed = 54321 + 1000 * trial

    if rank == 0:
        os.makedirs(f"lammps/{run}", exist_ok=True)

        erase = [
            log_dir,
            traj_dir,
            temp_dir,
            equilibrated_data_dir,
            contour_dir,
            mesh_traj_dir,
            mesh_distribution_dir
        ]

        for file_path in erase:
            if os.path.exists(file_path):
                os.remove(file_path)

    comm.Barrier()

    os.makedirs(f"lammps/{run}", exist_ok=True)

    L = lammps(comm=comm)

    L.command("clear")
    L.command(f"log {log_dir}")
    L.command("units lj")
    L.command("atom_style full")
    L.command("dimension 3")
    L.command("boundary p p p")
    L.command(f"read_data lammps/system{run}-np.data")
    L.command(f"mass 2 {str(np_mass)}")

    L.command("group polymer type 1")
    L.command("group nanoparticle type 2")
    L.command(f"group crosslinks id 1:{n_crosslink_nodes}")

    L.command("variable polymer_bead_count equal count(polymer)")

    polymer_bead_count = int(
        L.extract_variable("polymer_bead_count", None, 0)
    )

    if rank == 0:
        polymer_volume = polymer_bead_count*((4/3)*math.pi*0.5**3)
        polymer_fraction = polymer_volume / box_volume

    L.command("pair_style lj/cut 2.5")
    L.command("pair_modify shift yes")
    L.command(f"pair_coeff 1 1 1.0 1.0 {wca}")
    L.command(
        f"pair_coeff 1 2 1.0 {np_polymer_contact} "
        f"{wca * np_polymer_contact}"
    )
    L.command(
        f"pair_coeff 2 2 1.0 {np_np_contact} "
        f"{wca * np_np_contact}"
    )
    L.command("neigh_modify exclude type 2 2")

    L.command("bond_style harmonic")
    L.command("bond_coeff 1 25.0 1.0")
    L.command("special_bonds lj 0.0 0.0 0.0")

    L.command("minimize 1.0e-4 1.0e-6 1000 10000")

    L.command("fix 1 polymer nve")
    L.command(f"fix 2 polymer langevin 1.0 1.0 1.0 {polymer_seed}")
    L.command("fix 3 nanoparticle nve")
    L.command(f"fix 4 nanoparticle langevin 1.0 1.0 1.0 {np_seed}") #fix 4 nanoparticle langevin Tstart Tstop damp

    L.command("timestep 0.005")

    if float(amplitude) != 0.0:
        L.command("change_box all triclinic")
        L.command(
            f"fix shear all deform 1 xy wiggle $({amplitude}*ly) "
            f"{1 / frequency} remap x units box flip no"
        )
        L.command("variable gamma equal xy/ly")
    else:
        L.command("variable gamma equal 0.0")

    L.command("compute sanityTemp all temp")
    L.command("thermo_modify temp sanityTemp")

    #equilibration occurs here
    L.command("thermo 1000")
    L.command(
        "thermo_style custom step time temp press pe ke etotal v_gamma"
    )
    L.command(f"run {equilibration_steps}")

    L.command(f"write_data {equilibrated_data_dir}")
    comm.Barrier()

    if rank == 0:
        mean_Lc, std_Lc, min_Lc, max_Lc, n_strands = (
            measure_contour_lengths(
                equilibrated_data_dir,
                n_crosslink_nodes,
                contour_dir
            )
        )

        print(f"Trial {trial + 1} contour length:")
        print(f"  strands = {n_strands}")
        print(f"  mean Lc = {mean_Lc:.6f}")
        print(f"  std  Lc = {std_Lc:.6f}")
        print(f"  min  Lc = {min_Lc:.6f}")
        print(f"  max  Lc = {max_Lc:.6f}")

        strand_endpoints = find_strand_endpoints(
            equilibrated_data_dir,
            n_crosslink_nodes
        )
    else:
        strand_endpoints = None

    comm.Barrier()

    L.command("reset_timestep 0")

    L.command("variable current_step equal step")
    L.command("variable current_time equal time")

    L.command("compute polymerCOM polymer com")
    L.command("compute npCOM nanoparticle com")

    L.command("thermo 1000")
    L.command(
        "thermo_style custom step time temp press pe ke etotal v_gamma"
    )

    L.command(
        f"dump 1 nanoparticle custom 1000 {traj_dir} "
        f"id type mol xu yu zu ix iy iz"
    )
    L.command("dump_modify 1 sort id")

    L.command(
        f"dump 2 crosslinks custom 1000 {mesh_traj_dir} id xs ys zs"
    )
    L.command("dump_modify 2 sort id")

    L.command(
        f"fix tempOut all ave/time 1 1 1000 "
        f"v_current_time c_sanityTemp v_gamma "
        f"c_polymerCOM[1] c_polymerCOM[2] c_polymerCOM[3] "
        f"c_npCOM[1] c_npCOM[2] c_npCOM[3] "
        f"file {temp_dir} "
        f"title1 '# Temperature, shear, polymer COM, and NP COM output' "
        f"title2 '# step time temp gamma polymerCOM_x polymerCOM_y "
        f"polymerCOM_z npCOM_x npCOM_y npCOM_z'"
    )

    L.command(f"run {run_steps}")
    L.close()

    comm.Barrier()

    if rank == 0:
        mean_mesh_size, std_mesh_size, n_mesh_samples = (
            measure_mesh_size_distribution(
                mesh_traj_dir,
                strand_endpoints,
                mesh_distribution_dir
            )
        )

        print(f"Trial {trial + 1} mesh size distribution:")
        print(f"  samples = {n_mesh_samples}")
        print(f"  mean mesh size = {mean_mesh_size:.6f}")
        print(f"  std mesh size = {std_mesh_size:.6f}")

        with open(temp_dir, "a") as temp_file:
            temp_file.write("\n# Mesh size distribution statistics\n")
            temp_file.write(
                f"# Mesh size samples: {n_mesh_samples}\n"
            )
            temp_file.write(
                f"# Mean mesh size: {mean_mesh_size:.8f}\n"
            )
            temp_file.write(
                f"# Std mesh size: {std_mesh_size:.8f}\n"
            )

        os.remove(mesh_traj_dir)

    comm.Barrier()

    trial_elapsed_local = MPI.Wtime() - trial_start

    trial_elapsed = comm.reduce(
        trial_elapsed_local,
        op=MPI.MAX,
        root=0
    )

    if rank == 0:
        timing_dir = (
            f"lammps/{run}/timing{amplitude}_trial{trial + 1}.txt"
        )

        with open(timing_dir, "w") as timing_file:
            timing_file.write(f"trial: {trial + 1}\n")
            timing_file.write(f"mpi_processes: {n_ranks}\n")
            timing_file.write(
                f"elapsed_seconds: {trial_elapsed:.6f}\n"
            )
            timing_file.write(
                f"elapsed_minutes: {trial_elapsed / 60.0:.6f}\n"
            )
            timing_file.write(
                f"elapsed_hours: {trial_elapsed / 3600.0:.6f}\n"
            )

        print(
            f"Trial {trial + 1} finished in "
            f"{trial_elapsed:.2f} seconds "
            f"({trial_elapsed / 60.0:.2f} minutes)"
        )

comm.Barrier()
overall_elapsed_local = MPI.Wtime() - overall_start

overall_elapsed = comm.reduce(
    overall_elapsed_local,
    op=MPI.MAX,
    root=0
)

if rank == 0:
    overall_timing_dir = (
        f"lammps/{run}/timing{amplitude}_all_trials.txt"
    )

    with open(overall_timing_dir, "w") as timing_file:
        timing_file.write(f"trials: {trials}\n")
        timing_file.write(f"mpi_processes: {n_ranks}\n")
        timing_file.write(
            f"elapsed_seconds: {overall_elapsed:.6f}\n"
        )
        timing_file.write(
            f"elapsed_minutes: {overall_elapsed / 60.0:.6f}\n"
        )
        timing_file.write(
            f"elapsed_hours: {overall_elapsed / 3600.0:.6f}\n"
        )

    print(
        f"Finished all trials in {overall_elapsed:.2f} seconds "
        f"({overall_elapsed / 3600.0:.3f} hours)"
    )

    print(f"Polymer volume fraction: {polymer_fraction}")
