import os
from collections import namedtuple
import numpy as np


n_cells_per_dir = 6
mesh_size = 6.0
contour_length = 42
np_radius = 1.8
np_positions = [(21.0, 21.0, 21.0)]
out_dir = "lammps"

internal_beads_per_chain = contour_length - 1           # number_of_beads = contour_length - 1
sample_label = f"{contour_length}-6-np"


poly_type = 1
np_type = 2
mass_poly = 1.0
bond_type = 1
mass_np = (2.0 * np_radius) ** 3

Atom = namedtuple("Atom", ["id", "mol", "type", "q", "x", "y", "z"])
Bond = namedtuple("Bond", ["id", "type", "a", "b"])

def wrap(x, L):
    return x % L


def min_image(d, L):
    if d > 0.5 * L:
        d -= L
    elif d < -0.5 * L:
        d += L
    return d

def arc_positions_90(Apos, Bpos, N_internal, box_lengths, perp_dir):
    Lx, Ly, Lz = box_lengths
    disp = np.array([
        min_image(Bpos[0] - Apos[0], Lx),
        min_image(Bpos[1] - Apos[1], Ly),
        min_image(Bpos[2] - Apos[2], Lz),
    ], dtype=float)

    d = np.linalg.norm(disp)
    if d <= 0:
        raise ValueError("Distance between connected nodes is zero or negative.")

    e1 = disp / d
    e2 = np.array(perp_dir, dtype=float)
    e2 = e2 - np.dot(e2, e1) * e1
    e2_norm = np.linalg.norm(e2)
    if e2_norm < 1e-12:
        raise ValueError("perp_dir is parallel to the strand direction.")
    e2 = e2 / e2_norm

    theta = 0.5 * np.pi
    R = d / (2.0 * np.sin(theta / 2.0))
    M = Apos + 0.5 * disp
    h = R * np.cos(theta / 2.0)
    center = M - h * e2

    positions = []
    n_bonds = N_internal + 1
    for k in range(1, n_bonds):
        alpha = -theta / 2.0 + k * theta / n_bonds
        pos = center + R * (np.sin(alpha) * e1 + np.cos(alpha) * e2)
        positions.append(np.array([
            wrap(pos[0], Lx),
            wrap(pos[1], Ly),
            wrap(pos[2], Lz),
        ], dtype=float))

    return positions

def validate_inputs():
    if n_cells_per_dir < 2:
        raise ValueError("n_cells_per_dir must be at least 2.")
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive.")
    if internal_beads_per_chain < 0:
        raise ValueError("internal_beads_per_chain cannot be negative.")

    #remove the next two lines if generating a no-NP system
    if np_radius <= 0:
        raise ValueError("np_radius must be positive.")
    if not np_positions:
        raise ValueError("np_positions must contain at least one nanoparticle position.")

def main():
    validate_inputs()

    Lx = n_cells_per_dir * mesh_size
    Ly = n_cells_per_dir * mesh_size
    Lz = n_cells_per_dir * mesh_size
    box = (0.0, Lx, 0.0, Ly, 0.0, Lz)

    atoms = []
    bonds = []
    atom_id = 1
    mol_id = 1
    node_index = {}

    #Creates the lattice of crosslinks
    for iz in range(n_cells_per_dir):
        for iy in range(n_cells_per_dir):
            for ix in range(n_cells_per_dir):
                x = ix * mesh_size
                y = iy * mesh_size
                z = iz * mesh_size
                atoms.append(Atom(atom_id, mol_id, poly_type, 0.0, x, y, z))
                node_index[(ix, iy, iz)] = atom_id
                atom_id += 1
                mol_id += 1

    bond_id = 1

    for iz in range(n_cells_per_dir):
        for iy in range(n_cells_per_dir):
            for ix in range(n_cells_per_dir):
                A0 = node_index[(ix, iy, iz)]
                Apos = np.array([ix * mesh_size, iy * mesh_size, iz * mesh_size], dtype=float)

                Bx_idx = ((ix + 1) % n_cells_per_dir, iy, iz)
                B = node_index[Bx_idx]
                Bpos = np.array([Bx_idx[0] * mesh_size, Bx_idx[1] * mesh_size, Bx_idx[2] * mesh_size], dtype=float)
                sign = 1.0 if (iy + iz) % 2 == 0 else -1.0
                positions = arc_positions_90(Apos, Bpos, internal_beads_per_chain, (Lx, Ly, Lz), [0.0, sign, 0.0])
                A = A0
                for pos in positions:
                    x, y, z = pos
                    atoms.append(Atom(atom_id, mol_id, poly_type, 0.0, x, y, z))
                    bonds.append(Bond(bond_id, bond_type, A, atom_id))
                    bond_id += 1
                    A = atom_id
                    atom_id += 1
                    mol_id += 1
                bonds.append(Bond(bond_id, bond_type, A, B))
                bond_id += 1

                By_idx = (ix, (iy + 1) % n_cells_per_dir, iz)
                B = node_index[By_idx]
                Bpos = np.array([By_idx[0] * mesh_size, By_idx[1] * mesh_size, By_idx[2] * mesh_size], dtype=float)
                sign = 1.0 if (ix + iz) % 2 == 0 else -1.0
                positions = arc_positions_90(Apos, Bpos, internal_beads_per_chain, (Lx, Ly, Lz), [0.0, 0.0, sign])
                A = A0
                for pos in positions:
                    x, y, z = pos
                    atoms.append(Atom(atom_id, mol_id, poly_type, 0.0, x, y, z))
                    bonds.append(Bond(bond_id, bond_type, A, atom_id))
                    bond_id += 1
                    A = atom_id
                    atom_id += 1
                    mol_id += 1
                bonds.append(Bond(bond_id, bond_type, A, B))
                bond_id += 1

                Bz_idx = (ix, iy, (iz + 1) % n_cells_per_dir)
                B = node_index[Bz_idx]
                Bpos = np.array([Bz_idx[0] * mesh_size, Bz_idx[1] * mesh_size, Bz_idx[2] * mesh_size], dtype=float)
                sign = 1.0 if (ix + iy) % 2 == 0 else -1.0
                positions = arc_positions_90(Apos, Bpos, internal_beads_per_chain, (Lx, Ly, Lz), [sign, 0.0, 0.0])
                A = A0
                for pos in positions:
                    x, y, z = pos
                    atoms.append(Atom(atom_id, mol_id, poly_type, 0.0, x, y, z))
                    bonds.append(Bond(bond_id, bond_type, A, atom_id))
                    bond_id += 1
                    A = atom_id
                    atom_id += 1
                    mol_id += 1
                bonds.append(Bond(bond_id, bond_type, A, B))
                bond_id += 1

    np_mol_start = mol_id
    for k, (cx, cy, cz) in enumerate(np_positions):
        atoms.append(Atom(atom_id, np_mol_start + k, np_type, 0.0, wrap(cx, Lx), wrap(cy, Ly), wrap(cz, Lz)))
        atom_id += 1

    atoms = [Atom(a.id, a.mol, a.type, a.q, a.x % Lx, a.y % Ly, a.z % Lz) for a in atoms]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"system{sample_label}.data")

    with open(out_path, "w") as f:
        f.write("LAMMPS data file\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write("0 angles\n\n")
        f.write("2 atom types\n")
        f.write("1 bond types\n")
        f.write("0 angle types\n\n")
        f.write(f"{box[0]:.6f} {box[1]:.6f} xlo xhi\n")
        f.write(f"{box[2]:.6f} {box[3]:.6f} ylo yhi\n")
        f.write(f"{box[4]:.6f} {box[5]:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write(f"1 {mass_poly:.6f}\n")
        f.write(f"2 {mass_np:.6f}\n\n")
        f.write("Atoms # full\n\n")
        for a in atoms:
            f.write(f"{a.id} {a.mol} {a.type} {a.q:.6f} {a.x:.6f} {a.y:.6f} {a.z:.6f}\n")
        f.write("\nBonds\n\n")
        for b in bonds:
            f.write(f"{b.id} {b.type} {b.a} {b.b}\n")

    np_xyz = np.array([(a.x, a.y, a.z) for a in atoms if a.type == np_type], dtype=float)
    polymer_beads = sum(1 for a in atoms if a.type == poly_type)
    n_nodes = n_cells_per_dir ** 3
    n_strands = 3 * n_nodes

    print("Finished")
    print(f"Wrote: {out_path}")
    print(f"Box: {Lx:.6f} x {Ly:.6f} x {Lz:.6f}")
    print(f"Nodes: {n_nodes}")
    print(f"Strands: {n_strands}")
    print(f"Polymer beads: {polymer_beads}")
    print(f"Internal beads per chain: {internal_beads_per_chain}")
    print(f"NP radius: {np_radius}")
    print(f"NP mass: {mass_np:.6f}")
    print(f"Number of NPs: {len(np_xyz)}")
    print(f"NP positions:\n{np_xyz}")
    print(f"Total atoms: {len(atoms)}")
    print(f"Total bonds: {len(bonds)}")
    print("Total angles: 0")

if __name__ == "__main__":
    main()
