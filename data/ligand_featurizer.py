# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/4/30 14:19
@author: LiFan Chen
@Filename: mol_featurizer.py
@Software: PyCharm
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data

num_atom_feat = 40

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(13),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # 12-dim

    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency,dtype=np.float32)


def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    #mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


def smiles_to_data(smiles):
    atom_feat, adj_matrix = mol_features(smiles)
    edge_index = np.array(np.nonzero(adj_matrix), dtype=np.int64)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(atom_feat, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

if __name__ == "__main__":

    import os
    from tqdm import tqdm
    import torch
    import pickle
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser(description="Convert SMILES CSV to PyG dataset")

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to input CSV file (must contain 'Ligand_smiles' column)"
    )

    args = parser.parse_args()

    csv_path = args.csv_path

    base_dir = os.path.dirname(os.path.abspath(csv_path))
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(base_dir, f"{csv_stem}_ligand.pkl")

    df = pd.read_csv(csv_path)

    if 'Ligand_smiles' not in df.columns:
        raise ValueError("The specified column 'Ligand_smiles' does not exist in the CSV file.")

    smiles_list = df['Ligand_smiles'].dropna().tolist()
    N = len(smiles_list)
    print(f"Total entries: {N}")

    dataset = []


    for smiles in tqdm(smiles_list, total=N, desc="Processing"):
        try:
            data_obj = smiles_to_data(smiles)
            dataset.append(data_obj)
        except Exception as e:
            print(f"\n[ERROR] SMILES: {smiles}")
            print(e)

    os.makedirs(base_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {output_path}")