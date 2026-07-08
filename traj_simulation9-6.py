from lammps import lammps
import os
import math
from collections import defaultdict

n_cells_per_dir = 6
n_crosslink_nodes = n_cells_per_dir ** 3

L_c = f'9-{str(n_cells_per_dir)}'  #adjustable
amplitude = '0.00'  #adjustable
np_radius = 1.8
frequency = 0.002

#these are used for cutoffs
wca = 1.12246
np_polymer_contact = 0.5 + np_radius
np_np_contact = 2 * np_radius

trials = 10

equilibration_steps = 50_000
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

#this measures the true contour lengths after equilibration
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

    if Lx is None or Ly is None or Lz is None:
        raise ValueError("Could not read box dimensions from data file.")

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

                if len(next_candidates) != 1:
                    raise ValueError(
                        f"Unexpected topology at atom {current}: "
                        f"found {len(next_candidates)} forward neighbors."
                    )

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

for trial in range (trials):
    print(f"RUNNING TRIAL {trial + 1}")

    log_dir = f"lammps/{L_c}/log{amplitude}_trial{trial+1}.lammps"
    traj_dir = f"lammps/{L_c}/trajectory{amplitude}_trial{trial+1}.lammpstrj"
    temp_dir = f"lammps/{L_c}/temperature{amplitude}_trial{trial+1}.txt"
    equilibrated_data_dir = f"lammps/{L_c}/equilibrated{amplitude}_trial{trial + 1}.data"
    contour_dir = f"lammps/{L_c}/contour_lengths{amplitude}_trial{trial + 1}.txt"

    polymer_seed = 12345 + 1000 * trial
    np_seed = 54321 + 1000 * trial

    erase = [log_dir, traj_dir, temp_dir, equilibrated_data_dir, contour_dir]
    for f in erase:
        if os.path.exists(f):
            os.remove(f)

    os.makedirs(f"lammps/{L_c}", exist_ok=True)

    L = lammps()

    L.command("clear")
    L.command(f"log {log_dir}")
    L.command("units lj")
    L.command("atom_style full")
    L.command("dimension 3")
    L.command("boundary p p p")
    L.command(f"read_data lammps/system{L_c}-np.data")
    L.command(f"mass 2 {str(np_mass)}")

    L.command("group polymer type 1")
    L.command("group nanoparticle type 2")

    L.command("pair_style lj/cut 2.5")
    L.command("pair_modify shift yes")
    L.command(f"pair_coeff 1 1 1.0 1.0 {wca}")
    L.command(f"pair_coeff 1 2 1.0 {np_polymer_contact} {wca * np_polymer_contact}")
    L.command(f"pair_coeff 2 2 1.0 {np_np_contact} {wca * np_np_contact}")
    L.command("neigh_modify exclude type 2 2")

    L.command("bond_style harmonic")
    L.command("bond_coeff 1 25.0 1.0")
    L.command("special_bonds lj 0.0 0.0 0.0")

    L.command("minimize 1.0e-4 1.0e-6 1000 10000")

    L.command("fix 1 polymer nve")
    L.command(f"fix 2 polymer langevin 1.0 1.0 1.0 {polymer_seed}")
    L.command("fix 3 nanoparticle nve")
    L.command(f"fix 4 nanoparticle langevin 1.0 1.0 1.0 {np_seed}")

    L.command("timestep 0.005")

    if float(amplitude) != 0.0:
        L.command("change_box all triclinic")
        L.command(
            f"fix shear all deform 1 xy wiggle $({amplitude}*ly) {1 / frequency} "
            f"remap x units box flip no"
        )
        L.command("variable gamma equal xy/ly")
    else:
        L.command("variable gamma equal 0.0")

    L.command("compute sanityTemp all temp")
    L.command("thermo_modify temp sanityTemp")

    #equilibration occurs here
    L.command("thermo 1000")
    L.command("thermo_style custom step time temp press pe ke etotal v_gamma")
    L.command(f"run {equilibration_steps}")

    L.command(f"write_data {equilibrated_data_dir}")

    mean_Lc, std_Lc, min_Lc, max_Lc, n_strands = measure_contour_lengths(
        equilibrated_data_dir,
        n_crosslink_nodes,
        contour_dir
    )

    print(f"Trial {trial + 1} contour length:")
    print(f"  strands = {n_strands}")
    print(f"  mean Lc = {mean_Lc:.6f}")
    print(f"  std  Lc = {std_Lc:.6f}")
    print(f"  min  Lc = {min_Lc:.6f}")
    print(f"  max  Lc = {max_Lc:.6f}")

    L.command("reset_timestep 0")



    L.command("variable current_step equal step")
    L.command("variable current_time equal time")

    L.command("compute polymerCOM polymer com")
    L.command("compute npCOM nanoparticle com")

    L.command("thermo 1000")
    L.command("thermo_style custom step time temp press pe ke etotal v_gamma")

    L.command(f"dump 1 nanoparticle custom 1000 {traj_dir} id type mol xu yu zu ix iy iz")
    L.command("dump_modify 1 sort id")

    L.command(
        f"fix tempOut all ave/time 1 1 1000 "
        f"v_current_time c_sanityTemp v_gamma "
        f"c_polymerCOM[1] c_polymerCOM[2] c_polymerCOM[3] "
        f"c_npCOM[1] c_npCOM[2] c_npCOM[3] "
        f"file {temp_dir} "
        f"title1 '# Temperature, shear, polymer COM, and NP COM output' "
        f"title2 '# step time temp gamma polymerCOM_x polymerCOM_y polymerCOM_z npCOM_x npCOM_y npCOM_z'"
    )

    L.command(f"run {run_steps}")
    L.close()

    print(f"trial {trial+1} finished")

print("finished all trials")
