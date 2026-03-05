"""
Molecule → PyG Graph Featurizer
--------------------------------
Converts SMILES strings into PyTorch Geometric Data objects.
Each atom becomes a node with a feature vector.
Each bond becomes two directed edges.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


# ── Atom Feature Definitions ──────────────────────────────────────────────────

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Other']
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

NODE_FEATURE_DIM = 9   # must match model's node_features param


def atom_features(atom) -> list[float]:
    """
    9-dimensional atom feature vector:
      [0]   Atom type (one-hot index, normalized)
      [1]   Atomic number (normalized)
      [2]   Degree (number of bonds)
      [3]   Formal charge
      [4]   Num hydrogens
      [5]   Is in ring
      [6]   Is aromatic
      [7]   Hybridization (index, normalized)
      [8]   Chirality (0=none, 1=CW, 2=CCW)
    """
    symbol = atom.GetSymbol()
    atom_type_idx = ATOM_TYPES.index(symbol) if symbol in ATOM_TYPES else len(ATOM_TYPES) - 1

    hyb = atom.GetHybridization()
    hyb_idx = HYBRIDIZATION.index(hyb) if hyb in HYBRIDIZATION else 0

    chiral = atom.GetChiralTag()
    chiral_val = 0
    if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        chiral_val = 1
    elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        chiral_val = 2

    return [
        atom_type_idx / len(ATOM_TYPES),          # normalized atom type
        atom.GetAtomicNum() / 53.0,                # normalized atomic number (I=53)
        atom.GetDegree() / 6.0,                    # normalized degree
        atom.GetFormalCharge() / 2.0,              # normalized charge
        atom.GetTotalNumHs() / 4.0,                # normalized H count
        float(atom.IsInRing()),
        float(atom.GetIsAromatic()),
        hyb_idx / len(HYBRIDIZATION),              # normalized hybridization
        chiral_val / 2.0,                          # normalized chirality
    ]


# ── SMILES → PyG Data ─────────────────────────────────────────────────────────

def smiles_to_graph(smiles: str, label: float = None) -> Data | None:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: Input SMILES string
        label:  Target pChEMBL value (optional, for training)

    Returns:
        PyG Data object with:
            x           — node feature matrix [num_atoms, 9]
            edge_index  — COO edge list [2, num_bonds*2]
            edge_attr   — bond features [num_bonds*2, 4]
            y           — pChEMBL label (if provided)
            smiles      — original SMILES (for reference)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Node Features ─────────────────────────────────────────────────────────
    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)

    # ── Edge Index + Edge Features ────────────────────────────────────────────
    edge_indices = []
    edge_attrs = []

    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond_type_map.get(bond.GetBondType(), 0)

        bond_feat = [
            bt / 3.0,                              # bond type (normalized)
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
        ]

        # Add both directions (undirected graph)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [bond_feat, bond_feat]

    if not edge_indices:
        # Single atom molecule — add self loop
        edge_indices = [[0, 0]]
        edge_attrs = [[0.0, 0.0, 0.0, 0.0]]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # ── Morgan Fingerprint ────────────────────────────────────────────────────
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fingerprint = torch.tensor(list(fp), dtype=torch.float)

    # ── Assemble Data Object ──────────────────────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        fingerprint=fingerprint,
        smiles=smiles,
        num_nodes=x.size(0),
    )

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data


def smiles_list_to_graphs(smiles_list: list, labels: list = None) -> list[Data]:
    """
    Convert a list of SMILES to graphs, skipping invalid molecules.

    Args:
        smiles_list: List of SMILES strings
        labels:      Optional list of pChEMBL values

    Returns:
        List of valid PyG Data objects
    """
    graphs = []
    skipped = 0

    for i, smi in enumerate(smiles_list):
        label = labels[i] if labels is not None else None
        g = smiles_to_graph(smi, label)
        if g is not None:
            graphs.append(g)
        else:
            skipped += 1

    if skipped > 0:
        print(f"[Featurizer] Skipped {skipped} invalid SMILES out of {len(smiles_list)}")

    return graphs
