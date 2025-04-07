# Adapted from: https://github.com/gcorso/DiffDock

import ast
import binascii
import copy
import glob
import os
import pickle  # nosec
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from beartype.typing import Optional
from rdkit import Chem
from rdkit.Chem import AddHs, MolFromSmiles, RemoveAllHs, RemoveHs
from torch_geometric.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import plotly.graph_objects as go


from pytorch3d.ops import corresponding_points_alignment
from flowdock.utils.frame_utils import apply_similarity_transform
from flowdock.data.components.mol_features import mol_to_graph
import networkx as nx


from sympy.combinatorics import Permutation, PermutationGroup

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components.mol_features import process_molecule
from flowdock.data.components.process_mols import generate_conformer, read_molecule
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    centralize_complex_graph,
    centralize_complex_graph_numpy,
    merge_protein_and_ligands,
    pdb_filepath_to_protein,
    process_protein,
    align_apo_to_holo,
)
from flowdock.utils.model_utils import sample_inplace_to_torch
from flowdock.utils.utils import read_strings_from_txt
import pynauty
from collections import deque
import networkx as nx 

PROTEIN_CUTOFF = 1.0
LIGAND_CUTOFF = 2.0

log = RankedLogger(__name__, rank_zero_only=True)

color_list = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'brown', 'black', 'gray', 'cyan', 'magenta', 'olive', 'navy', 'lime', 'teal', 'aqua', 'maroon', 'fuchsia', 'silver', 'white']

def save_3d_graph(G, ligand_graph=None, new_ligand_graph=None, binding_site_nodes=None, isomorphisms=None, filename="protein_ligand_graph.html", only_binding_site=False, only_molecule=False):
    """
    Saves a 3D visualization of the protein Cα graph, highlighting binding site nodes, and including ligand graph.
    
    Args:
        G (nx.Graph): The Cα graph.
        ligand_graph (nx.Graph): The ligand graph (optional).
        binding_site_nodes (list): List of node indices belonging to the binding site.
        filename (str): Path to save the HTML file.
    """
    edge_x, edge_y, edge_z = [], [], []
    edge_colors = []

    binding_site_nodes = set(binding_site_nodes) if binding_site_nodes else set()
    
    iso_nodes = list()
    for iso in isomorphisms:
        iso_nodes.extend(list(iso.keys()))
    iso_nodes = set(iso_nodes)
    binding_site_nodes = binding_site_nodes.union(iso_nodes)

    # Get protein graph edges coordinates
    if not only_molecule:
        for edge in G.edges():
            coord1, coord2 = G.nodes[edge[0]]["pos"], G.nodes[edge[1]]["pos"]
            # Highlight edges connecting binding site nodes in green
            if edge[0] in binding_site_nodes and edge[1] in binding_site_nodes:
                edge_colors.append("green")
                edge_x.extend([coord1[0], coord2[0], None])
                edge_y.extend([coord1[1], coord2[1], None])
                edge_z.extend([coord1[2], coord2[2], None])
            elif not only_binding_site:
                edge_colors.append("blue")
                edge_x.extend([coord1[0], coord2[0], None])
                edge_y.extend([coord1[1], coord2[1], None])
                edge_z.extend([coord1[2], coord2[2], None])

    # Get ligand graph edges coordinates
    if ligand_graph:
        for edge in ligand_graph.edges():
            coord1, coord2 = ligand_graph.nodes[edge[0]]["pos"], ligand_graph.nodes[edge[1]]["pos"]
            edge_x.extend([coord1[0], coord2[0], None])
            edge_y.extend([coord1[1], coord2[1], None])
            edge_z.extend([coord1[2], coord2[2], None])
            edge_colors.append("purple")  # Color edges in ligand graph as purple
    
    if not only_molecule:
        if new_ligand_graph:
            for edge in new_ligand_graph.edges():
                coord1, coord2 = new_ligand_graph.nodes[edge[0]]["pos"], new_ligand_graph.nodes[edge[1]]["pos"]
                edge_x.extend([coord1[0], coord2[0], None])
                edge_y.extend([coord1[1], coord2[1], None])
                edge_z.extend([coord1[2], coord2[2], None])
                edge_colors.append("orange")

    # Node positions for the protein graph
    node_x, node_y, node_z = [], [], []
    node_colors = []

    if not only_molecule:
        for i in G.nodes():
            x, y, z = G.nodes[i]["pos"]
            
            # Highlight binding site nodes in green, others in red
            if i in binding_site_nodes:
                node_colors.append("green")
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
            elif not only_binding_site:
                node_colors.append("red")
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)

    # Node positions for the ligand graph
    if ligand_graph:
        color_dict = dict()
        color_idx = -1
        for i in ligand_graph.nodes():
            x, y, z = ligand_graph.nodes[i]["pos"]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            color = ligand_graph.nodes[i]["node_type"]
            
            if color not in color_dict:
                color_idx += 1
                color_dict[color] = color_idx
                node_colors.append(color_list[color_dict[color]]) # add 0
            else:
                node_colors.append(color_list[color_dict[color]])
                
    
    if not only_molecule:
        if new_ligand_graph:
            for i in new_ligand_graph.nodes():
                x, y, z = new_ligand_graph.nodes[i]["pos"]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                node_colors.append("blue")

    # Create Plotly figure
    fig = go.Figure()

    # Add protein graph edges
    for i in range(len(edge_x) // 3):  # Each edge has 3 elements (x1, x2, None)
        fig.add_trace(go.Scatter3d(
            x=edge_x[i * 3: (i + 1) * 3], 
            y=edge_y[i * 3: (i + 1) * 3], 
            z=edge_z[i * 3: (i + 1) * 3], 
            mode='lines', 
            line=dict(color=edge_colors[i], width=2)
        ))

    # Add ligand graph edges (if any)
    if ligand_graph:
        for i in range(len(edge_x) // 3, len(edge_x) // 3 + len(ligand_graph.edges())):
            fig.add_trace(go.Scatter3d(
                x=edge_x[i * 3: (i + 1) * 3], 
                y=edge_y[i * 3: (i + 1) * 3], 
                z=edge_z[i * 3: (i + 1) * 3], 
                mode='lines', 
                line=dict(color='purple', width=2)
            ))

    # Add nodes (protein + ligand)
    fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, 
                               mode='markers', marker=dict(size=5, color=node_colors)))

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      title="3D Protein and Ligand Graph")

    # Save as HTML
    if only_binding_site:
        filename = filename.replace(".html", "_binding_site.html")
    fig.write_html(filename)
    print(f"Graph saved as {filename}")



def save_3d_graph_(ligand_graph=None, filename="ligand_graph.html"):
    """
    Saves a 3D visualization of the protein Cα graph, highlighting binding site nodes, and including ligand graph.
    
    Args:
        ligand_graph (nx.Graph): The ligand graph (optional).
        filename (str): Path to save the HTML file.
    """
    edge_x, edge_y, edge_z = [], [], []
    edge_colors = []    

    # Get ligand graph edges coordinates
    for edge in ligand_graph.edges():
        coord1, coord2 = ligand_graph.nodes[edge[0]]["pos"], ligand_graph.nodes[edge[1]]["pos"]
        edge_x.extend([coord1[0], coord2[0], None])
        edge_y.extend([coord1[1], coord2[1], None])
        edge_z.extend([coord1[2], coord2[2], None])
        edge_colors.append("purple")  # Color edges in ligand graph as purple
    
    # Node positions for the protein graph
    node_x, node_y, node_z = [], [], []
    node_colors = []


    # Node positions for the ligand graph
    color_dict = dict()
    color_idx = -1
    for i in ligand_graph.nodes():
        x, y, z = ligand_graph.nodes[i]["pos"]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        color = ligand_graph.nodes[i]["node_type"]
        
        if color not in color_dict:
            color_idx += 1
            color_dict[color] = color_idx
            node_colors.append(color_list[color_dict[color]]) # add 0
        else:
            node_colors.append(color_list[color_dict[color]])
            

    # Create Plotly figure
    fig = go.Figure()

    # Add protein graph edges
    for i in range(len(edge_x) // 3):  # Each edge has 3 elements (x1, x2, None)
        fig.add_trace(go.Scatter3d(
            x=edge_x[i * 3: (i + 1) * 3], 
            y=edge_y[i * 3: (i + 1) * 3], 
            z=edge_z[i * 3: (i + 1) * 3], 
            mode='lines', 
            line=dict(color=edge_colors[i], width=2)
        ))

    # Add ligand graph edges (if any)
    if ligand_graph:
        for i in range(len(edge_x) // 3, len(edge_x) // 3 + len(ligand_graph.edges())):
            fig.add_trace(go.Scatter3d(
                x=edge_x[i * 3: (i + 1) * 3], 
                y=edge_y[i * 3: (i + 1) * 3], 
                z=edge_z[i * 3: (i + 1) * 3], 
                mode='lines', 
                line=dict(color='purple', width=2)
            ))

    # Add nodes (protein + ligand)
    fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, 
                               mode='markers', marker=dict(size=5, color=node_colors)))

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      title="3D Graph")

    fig.write_html(filename)
    print(f"Graph saved as {filename}")


def check_identity_dict(dt):
    ret = True
    for key in dt:
        if key != dt[key]:
            ret = False
            break
    return ret
    

def build_ca_graph(ca_coords, cutoff=PROTEIN_CUTOFF, node_type=None):
    """
    Constructs a graph where nodes are Cα atoms and edges exist if two atoms are within a cutoff distance.

    Args:
        ca_coords (torch.Tensor): (N, 3) tensor of Cα atomic coordinates.
        cutoff (float): Distance threshold for connecting nodes (in Angstroms).
        node_type (torch.Tensor, optional): (N,) tensor indicating node types.

    Returns:
        nx.Graph: The generated protein Cα graph.
    """
    num_atoms = ca_coords.shape[0]
    G = nx.Graph()

    # Compute pairwise distances efficiently using torch.cdist()
    dist_matrix = torch.cdist(ca_coords, ca_coords)  # (N, N)

    # Get indices where distance is within cutoff (excluding self-connections)
    edge_indices = torch.nonzero((dist_matrix <= cutoff) & (dist_matrix > 0), as_tuple=False)

    # Add nodes
    for i in range(num_atoms):
        node_data = {'pos': ca_coords[i].tolist()}
        if node_type is not None:
            if isinstance(node_type, torch.Tensor):
                node_data['node_type'] = int(node_type[i].item())  # Convert tensor value to int
            else:
                node_data['node_type'] = node_type[i]
        G.add_node(i, **node_data)

    # Add edges
    for i, j in edge_indices:
        G.add_edge(int(i.item()), int(j.item()))
        G.add_edge(int(j.item()), int(i.item()))  # Undirected graph

    return G

def build_knn_graph(ca_coords, k=5, node_type=None):
    """
    Constructs a k-Nearest Neighbors (kNN) graph where nodes are Cα atoms and each node connects to its k nearest neighbors.

    Args:
        ca_coords (torch.Tensor): (N, 3) tensor of Cα atomic coordinates.
        k (int): Number of nearest neighbors to connect for each node.
        node_type (torch.Tensor, optional): (N,) tensor indicating node types.

    Returns:
        nx.Graph: The generated protein Cα kNN graph.
    """
    num_atoms = ca_coords.shape[0]
    G = nx.Graph()

    # Compute pairwise distances using torch.cdist()
    dist_matrix = torch.cdist(ca_coords, ca_coords)  # (N, N)
    
    # Get the indices of the k nearest neighbors for each node
    knn_indices = torch.argsort(dist_matrix, dim=1)[:, 1:k+1]  # Exclude self (index 0)

    # Add nodes
    for i in range(num_atoms):
        node_data = {'pos': ca_coords[i].tolist()}
        if node_type is not None:
            if isinstance(node_type, torch.Tensor):
                node_data['node_type'] = int(node_type[i].item())
            else:
                node_data['node_type'] = node_type[i]
        G.add_node(i, **node_data)
    
    # Add edges
    for i in range(num_atoms):
        for j in knn_indices[i]:
            G.add_edge(int(i), int(j.item()))  # Undirected graph
            G.add_edge(int(j.item()), int(i))
    
    return G



def find_binding_site(residue_coords, ligand_pos, res_atom_mask):
    '''
    residue_coords: torch.Tensor, shape (N, 3)
    ligand_pos: torch.Tensor, shape (M, 3)
    res_atom_mask: torch.Tensor, shape (N,)
    
    Returns:
    set of residue indices that are within 6.0 Å of any ligand atom.
    '''
    # Filter masked residues
    residue_coords = residue_coords[res_atom_mask.bool()]  # (N', 3)
    
    # Compute pairwise distances efficiently using broadcasting
    dists = torch.cdist(residue_coords, ligand_pos)  # (N', M)
    
    # Find indices where distance < 6.0 Å
    binding_residue_mask = (dists < 6.0).any(dim=1)  # (N',)
    
    # Convert back to original indices
    valid_indices = torch.nonzero(res_atom_mask, as_tuple=True)[0]  # Original indices
    
    # Compute residue indices (i // 37) and return unique set
    binding_residues = torch.div(valid_indices[binding_residue_mask], 37, rounding_mode='floor')
    
    return set(binding_residues.tolist())


def match_amino_acids(node1, node2):
    """
    Custom node comparison function to match amino acid types during subgraph isomorphism.
    
    Args:
        node1: Node in the first graph (target graph).
        node2: Node in the second graph (subgraph).
        
    Returns:
        True if the amino acid types of the nodes match, otherwise False.
    """
    return node1["node_type"] == node2["node_type"]


def find_isomorphism_bindingsite(G, binding_site_graph):
    """
    Finds the subgraph isomorphisms between the protein graph and ligand graph.
    
    Args:
        G (nx.Graph): The protein graph.
        ligand_graph (nx.Graph): The ligand graph.
        
    Returns:
        list: List of subgraph isomorphisms.
    """
    # Find subgraph isomorphisms
    GM = nx.algorithms.isomorphism.GraphMatcher(G, binding_site_graph, node_match=match_amino_acids)
    isomorphisms = list(GM.subgraph_isomorphisms_iter())
    return isomorphisms


class PDBBindDataset(Dataset):
    """A PyTorch Geometric Dataset for PDBBind dataset."""

    def __init__(
        self,
        root,
        transform=None,
        cache_path=os.path.join("data", "cache"),
        split_path="data" + os.sep,
        limit_complexes=0,
        num_workers=0,
        max_lig_size=None,
        remove_hs=False,
        num_conformers=1,
        esm_embeddings_path=None,
        apo_protein_structure_dir=None,
        require_ligand=False,
        include_miscellaneous_atoms=False,
        protein_path_list=None,
        ligand_descriptions=None,
        keep_local_structures=False,
        protein_file="protein_processed",
        ligand_file="ligand",
        min_protein_length: Optional[int] = 10,
        max_protein_length: Optional[int] = 4000,
        is_test_dataset=False,
        a2h_assessment_csv_filepath=None,
        filter_using_a2h_assessment=False,
        a2h_min_tmscore=None,
        a2h_max_rmsd=None,
        a2h_min_protein_length=None,
        a2h_max_protein_length=None,
        a2h_min_ligand_length=None,
        a2h_max_ligand_length=None,
        binding_affinity_values_dict=None,
        n_lig_patches=32,
        overfit_example_name=None,
    ):
        """Initializes the dataset."""

        super().__init__(root, transform)
        self.pdbbind_dir = root
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes # 0
        self.num_workers = num_workers
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.apo_protein_structure_dir = apo_protein_structure_dir
        self.use_old_wrong_embedding_order = False
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.protein_file = protein_file
        self.ligand_file = ligand_file
        self.min_protein_length = min_protein_length
        self.max_protein_length = max_protein_length
        self.is_test_dataset = is_test_dataset
        self.binding_affinity_values_dict = binding_affinity_values_dict
        self.n_lig_patches = n_lig_patches
        self.overfit_example_name = overfit_example_name

        split = os.path.splitext(os.path.basename(self.split_path))[0]
        self.full_cache_path = os.path.join(
            cache_path,
            f"PDBBind_limit{self.limit_complexes}"
            f"_INDEX{split}"
            f"_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}"
            + ("" if self.esm_embeddings_path is None else "_esmEmbeddings")
            + "_full"
            + ("" if not keep_local_structures else "_keptLocalStruct")
            + (
                ""
                if protein_path_list is None or ligand_descriptions is None
                else str(binascii.crc32("".join(ligand_descriptions + protein_path_list).encode()))
            )
            + ("" if protein_file == "protein_processed" else "_" + protein_file)
            + ("" if not self.include_miscellaneous_atoms else "_miscAtoms")
            + ("" if self.use_old_wrong_embedding_order else "_chainOrd")
            + ("" if min_protein_length is None else f"_minProteinLength{min_protein_length}")
            + ("" if max_protein_length is None else f"_maxProteinLength{max_protein_length}"),
        )
        
        vinh_cache_path = f"data/pdbbind/vinh_cache/{split}"
        vinh_ligand_cache_path = f"{vinh_cache_path}/ligands"
        vinh_complex_cache_path = f"{vinh_cache_path}/complexes"
        os.makedirs(vinh_ligand_cache_path, exist_ok=True)
        os.makedirs(vinh_complex_cache_path, exist_ok=True)
        
        all_files = os.listdir(vinh_complex_cache_path)
        if 1:
            self.num_conformers = num_conformers

            if not self.check_all_complexes():
                os.makedirs(self.full_cache_path, exist_ok=True)
                if protein_path_list is None or ligand_descriptions is None: # yes
                    self.preprocessing()
                else:
                    self.inference_preprocessing()

            self.complex_graphs, self.rdkit_ligands = self.collect_all_complexes()

            # analyze and potentially filter the PDBBind dataset based on its apo-to-holo (a2h) structural assessment
            if a2h_assessment_csv_filepath is not None and os.path.exists(a2h_assessment_csv_filepath):
                a2h_assessment_df = pd.read_csv(a2h_assessment_csv_filepath)
                a2h_assessment_df["ID"] = [
                    Path(paths[0]).stem[:4]
                    for paths in a2h_assessment_df["Filepath"].apply(ast.literal_eval).tolist()
                ]
                ligand_num_atoms = [
                    [int(num_atoms) for num_atoms in num_atoms_str.split(",")]
                    for num_atoms_str in a2h_assessment_df["Ligand_Num_Atoms"].tolist()
                ]
                a2h_assessment_df["Ligand_Total_Num_Atoms"] = np.array(
                    [np.array(num_atoms).sum() for num_atoms in ligand_num_atoms]
                )
                # import matplotlib.pyplot as plt
                # import seaborn as sns
                # plot_dir = Path(a2h_assessment_csv_filepath).parent / "plots"
                # plot_dir.mkdir(exist_ok=True)
                # plt.clf()
                # sns.histplot(a2h_assessment_df["TM-score"])
                # plt.title("Apo-To-Holo Protein TM-score")
                # plt.savefig(plot_dir / "a2h_TM-score_hist.png")
                # plt.clf()
                # sns.histplot(a2h_assessment_df["RMSD"])
                # plt.title("Apo-To-Holo Protein RMSD")
                # plt.savefig(plot_dir / "a2h_RMSD_hist.png")
                # plt.clf()
                # plt.xlim(0, 1500)
                # sns.histplot(a2h_assessment_df["Apo_Length"])
                # plt.title("Apo Protein Length")
                # plt.savefig(plot_dir / "apo_length_hist.png")
                # plt.clf()
                # plt.xlim(0, 500)
                # sns.histplot(a2h_assessment_df["Ligand_Total_Num_Atoms"])
                # plt.title("Ligand Total Number of Atoms")
                # plt.savefig(plot_dir / "ligand_total_num_atoms_hist.png")
                if filter_using_a2h_assessment and not is_test_dataset:
                    log.info(
                        f"Filtering the PDBBind {split} dataset based on its apo-to-holo (a2h) structural assessment"
                    )
                    a2h_assessment_df = a2h_assessment_df[
                        (a2h_assessment_df["TM-score"] >= a2h_min_tmscore)
                        & (a2h_assessment_df["RMSD"] <= a2h_max_rmsd)
                        & (a2h_assessment_df["Apo_Length"] >= a2h_min_protein_length)
                        & (a2h_assessment_df["Apo_Length"] <= a2h_max_protein_length)
                        & (a2h_assessment_df["Ligand_Total_Num_Atoms"] >= a2h_min_ligand_length)
                        & (a2h_assessment_df["Ligand_Total_Num_Atoms"] <= a2h_max_ligand_length)
                    ]
                    a2h_filtered_ids = {id: None for id in a2h_assessment_df["ID"].tolist()}
                    new_complex_graphs, new_rdkit_ligands = [], []
                    for complex_id, complex_obj in enumerate(self.complex_graphs):
                        if complex_obj["metadata"]["sample_ID"].lower() in a2h_filtered_ids:
                            new_complex_graphs.append(complex_obj)
                            new_rdkit_ligands.append(self.rdkit_ligands[complex_id])
                    self.complex_graphs = new_complex_graphs # 12993 # 11 # 2q55 # deterministics
                    self.rdkit_ligands = new_rdkit_ligands # 12993 - list
                    
                    # here I think we can already zero centered them! once

            list_names = [complex_obj["metadata"]["sample_ID"] for complex_obj in self.complex_graphs] # name of all proteins
            log.info(
                f"{len(list_names)} total complexes available from {self.full_cache_path} after all {split} filtering"
            )
            with open(
                os.path.join(
                    self.full_cache_path,
                    f"pdbbind_{os.path.splitext(os.path.basename(self.split_path))[0][:3]}_names.txt",
                ),
                "w",
            ) as f:
                f.write("\n".join(list_names))
                
            # OK! Zero centering the complexes
            log.info("Zero centering the complexes")
            
            
            per_sample_keys = [key for key in self.complex_graphs[0]['metadata'].keys() if "per_sample" in key]
            
            for i in tqdm(range(len(self.complex_graphs))):
                self.complex_graphs[i] = centralize_complex_graph_numpy(self.complex_graphs[i])
                # remove per_sample keys 
                for key in per_sample_keys:
                    del self.complex_graphs[i]['metadata'][key]
                # complex_id = self.complex_graphs[i]["metadata"]["sample_ID"]
                # save the zero centered complexes
                # with open(os.path.join(vinh_complex_cache_path, f"{complex_id}_complex.pkl"), "wb") as f:
                #     pickle.dump((self.complex_graphs[i]), f)
                # # save the ligands
                # with open(os.path.join(vinh_ligand_cache_path, f"{complex_id}_ligand.pkl"), "wb") as f:
                #     pickle.dump((self.rdkit_ligands[i]), f)
        # else:
        #     log.info(f"Loading complexes from {vinh_cache_path}")
        #     self.complex_graphs = []
        #     self.rdkit_ligands = []
        #     complexes = os.listdir(vinh_complex_cache_path)
            
        #     for complex_path in tqdm(complexes):
        #         with open(os.path.join(vinh_complex_cache_path, complex_path), "rb") as f:
        #             complex_graph = pickle.load(f)
        #             self.complex_graphs.append(complex_graph)
        #         ligand_path = complex_path.replace("_complex.pkl", "_ligand.pkl")
        #         with open(os.path.join(vinh_ligand_cache_path, ligand_path), "rb") as f:
        #             ligand = pickle.load(f)
        #             self.rdkit_ligands.append(ligand)    
            
        #     log.info(f"Loaded {len(self.complex_graphs)} complexes from {vinh_cache_path}")
        
        
        # now, we need to write a function to find the binding site of the protein
        # we can use the ligand to find the binding site
        # for i in range(len(self.complex_graphs)):
        #     complex_graph = self.complex_graphs[i]
        self.max_perm = 100
        # self.get(0, visualize=True)
        
        
    def len(self):
        """Returns the number of complexes in the dataset."""
        return len(self.complex_graphs)
    
    
    def assign_node_colors(self, edge_index, edge_attribute, node_attribute):
        """
        Constructs a dictionary mapping each node to its list of neighbors.
        
        Args:
            edge_index (torch.Tensor): A (2, num_edges) tensor representing the graph edges.

        Returns:
            dict: A dictionary where keys are node indices and values are lists of neighbors.
        """
        num_nodes = node_attribute.shape[0]
        num_edge_types = edge_attribute.shape[1]

        # Initialize aggregated edge attributes for each node
        node_edge_info = torch.zeros((num_nodes, num_edge_types), device=edge_attribute.device)

        # Aggregate edge attributes into node attributes
        node_edge_info.index_add_(0, edge_index[0], edge_attribute)
        node_edge_info.index_add_(0, edge_index[1], edge_attribute)  # Undirected case

        # Convert to binary presence (1 if the node has at least one connection of a given type)
        node_edge_info = (node_edge_info > 0).int()

        # Convert to tuple for hashing
        node_hash_values = torch.tensor(
            [hash((int(node_attr), tuple(edge_types.tolist()))) for node_attr, edge_types in zip(node_attribute, node_edge_info)],
            dtype=torch.int64,
            device=edge_attribute.device
        )
        return node_hash_values
    

    def get(self, idx, default_ligand_ccd_id: str = "XXX", visualize=False):
        """Returns a HeteroData object for a given index."""
        
        complex_graph = sample_inplace_to_torch(copy.deepcopy(self.complex_graphs[idx]))
        if self.require_ligand: # False
            complex_graph["metadata"]["mol"] = RemoveAllHs(copy.deepcopy(self.rdkit_ligands[idx]), sanitize=False)
        if self.binding_affinity_values_dict is not None:
            try:
                complex_graph["features"]["affinity"] = torch.tensor(
                    [
                        self.binding_affinity_values_dict[
                            complex_graph["metadata"]["sample_ID"].lower()
                        ][default_ligand_ccd_id]
                    ],
                    dtype=torch.float32,
                )
            except Exception as e:
                log.info(
                    f"Binding affinity value not found for {complex_graph['metadata']['sample_ID']} due to: {e}"
                )
                complex_graph["features"]["affinity"] = torch.tensor(
                    [torch.nan], dtype=torch.float32
                )
        else:
            complex_graph["features"]["affinity"] = torch.tensor([torch.nan], dtype=torch.float32)
        # align apo to holo
        # try:
        complex_graph["metadata"]["mol"] = RemoveAllHs(copy.deepcopy(self.rdkit_ligands[idx]), sanitize=False)
        mol = complex_graph["metadata"]["mol"]
        atomic_numbers = complex_graph["features"]["atomic_numbers"].long()
        num_nodes = len(atomic_numbers)
        indexer = complex_graph["indexer"]
        edges_ori = torch.stack((indexer["gather_idx_ij_i"],indexer["gather_idx_ij_j"]))
        edges = edges_ori[:, edges_ori[0] <= edges_ori[1]]  # de-duplicate edges
        graph = pynauty.Graph(num_nodes, directed=False)
        for i in range(edges.shape[1]):
            graph.connect_vertex(edges[0, i].item(), [edges[1, i].item()])
        
        edge_attributes_ori = torch.zeros(edges_ori.shape[1], 4)
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            first_atom = bond.GetBeginAtomIdx()
            second_atom = bond.GetEndAtomIdx()
            
            # find_index in the edges_ori
            bond_type_dict = {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3,
            }
            
            find_index = torch.where((edges_ori[0] == first_atom) & (edges_ori[1] == second_atom))[0]
            try:
                edge_attributes_ori[find_index[0],bond_type_dict[bond_type]] = 1
            except:
                # if the bond is not in the edge list, we need to add it
                edge_attributes_ori[find_index[0],0] = 1
            
            find_index = torch.where((edges_ori[0] == second_atom) & (edges_ori[1] == first_atom))[0]
            try:
                edge_attributes_ori[find_index[0],bond_type_dict[bond_type]] = 1
            except:
                # if the bond is not in the edge list, we need to add it
                edge_attributes_ori[find_index[0],0] = 1
            
        edge_attributes = edge_attributes_ori[edges_ori[0] < edges_ori[1]]
        edge_attributes = edge_attributes.float()
        colors = self.assign_node_colors(edges, edge_attributes, atomic_numbers).tolist()
        
        # fix the original code's edge attribute bug:
        
        num_atoms = complex_graph["metadata"]["num_i"]
        adjacency_mat = torch.zeros((num_atoms, num_atoms)).long()
        adjacency_mat[complex_graph["indexer"]["gather_idx_ij_i"],complex_graph["indexer"]["gather_idx_ij_j"]] = 1
        sum_pair_path_dist = [torch.eye(num_atoms).long()]
        for path_length in range(3):
            sum_pair_path_dist.append(torch.matmul(sum_pair_path_dist[-1], adjacency_mat))
        sum_pair_path_dist = torch.stack(sum_pair_path_dist, dim=2)
        atom_pair_feature_mat = torch.zeros((num_atoms, num_atoms, 4))
        atom_pair_feature_mat[edges_ori[0], edges_ori[1]] = edge_attributes_ori
        
        atom_pair_feature_mat = torch.cat(
            [atom_pair_feature_mat, (sum_pair_path_dist > 0).float()], dim=2
        )
        uv_adj_mat = torch.sum(sum_pair_path_dist, dim=2) > 0
        
        num_key_frames = complex_graph["metadata"]["num_U"]
        num_triplets = complex_graph["metadata"]["num_ijk"]
        key_atom_idx = complex_graph["indexer"]["gather_idx_U_u"]
        
        atom_frame_pair_feat_initial_ = torch.cat(
            [
                atom_pair_feature_mat[key_atom_idx, :][:, complex_graph["indexer"]["gather_idx_ijk_i"]],
                atom_pair_feature_mat[key_atom_idx, :][:, complex_graph["indexer"]["gather_idx_ijk_j"]],
                atom_pair_feature_mat[key_atom_idx, :][:, complex_graph["indexer"]["gather_idx_ijk_k"]],
            ],
            dim=2,
        ).reshape((num_key_frames * num_triplets, atom_pair_feature_mat.shape[2] * 3))
        
        complex_graph["features"]["atom_frame_pair_encodings"] = atom_frame_pair_feat_initial_
        complex_graph["features"]["bond_encodings"] = edge_attributes_ori
        complex_graph["features"]["atom_pair_encodings"] = atom_pair_feature_mat[uv_adj_mat]

        # finish fixing the bug
        
        color_groups = defaultdict(set)
        for node, color in enumerate(colors):
            color_groups[color].add(node)
        vertex_colors = list(color_groups.values())
        graph.set_vertex_coloring(vertex_colors)
        
        generators, order, o2, orbits, orbit_no = pynauty.autgrp(graph)
        
        # Convert generators to sympy Permutations
        generators_sympy = [Permutation(g) for g in generators]

        # Construct the permutation group
        aut_group = PermutationGroup(generators_sympy)

        # Print all isomorphisms (automorphisms)
        all_perms = []
        for perm in aut_group.generate():
            all_perms.append(perm.array_form)
        if len(all_perms) == 0:
            all_perms = [list(range(num_atoms))]
        
        if len(all_perms[0]) == 0:
            all_perms = [list(range(num_atoms))]
        
        
        all_perms = torch.stack([torch.tensor(perm) for perm in all_perms], dim=0).long()
        num_perms = all_perms.size(0)
        
        if num_perms > self.max_perm:
            # randomly sample max_perm permutations
            identity = all_perms[0].clone()
            all_perms = all_perms[torch.randperm(num_perms)]
            all_perms = all_perms[:self.max_perm]
            all_perms[0] = identity
            num_perms = self.max_perm
            
        complex_graph["features"]["all_perms"] = all_perms.t()
        complex_graph["metadata"]["num_perms"] = num_perms
        # complex_graph["metadata"]["max_perm"] = self.max_perm
        
        apo_res_atom_positions = complex_graph['features']['apo_res_atom_positions']
        res_atom_positions = complex_graph['features']['res_atom_positions']
        
        res_atom_mask = complex_graph["features"]["res_atom_mask"]
        apo_res_atom_mask = complex_graph["features"]["apo_res_atom_mask"]
        
        complex_graph["features"]["res_atom_mask"] = res_atom_mask * apo_res_atom_mask
        complex_graph["features"]["apo_res_atom_mask"] = res_atom_mask * apo_res_atom_mask
    
        res_atom_mask = complex_graph["features"]["res_atom_mask"]
        res_type = complex_graph["features"]["res_type"]
        #### FIND BINDING SITE ####
        vinh = False
        if vinh:
            res_atom_positions = complex_graph['features']['res_atom_positions']
            
            ligand_pos = complex_graph['features']['sdf_coordinates']
            res_atom_mask = complex_graph['features']['res_atom_mask']
            res_type = complex_graph['features']['res_type']
            residue_coords = res_atom_positions[:, 1]
            apo_residue_coords = apo_res_atom_positions[:, 1]
            
            flat_res_atom = res_atom_positions.reshape(-1, 3)
            flat_mask = res_atom_mask.reshape(-1)
            
            binding_site = find_binding_site(flat_res_atom, ligand_pos, flat_mask)
            
            # G = build_ca_graph(apo_residue_coords, cutoff=PROTEIN_CUTOFF, node_type=res_type)
            G = build_knn_graph(apo_residue_coords, k=5, node_type=res_type)
            
            binding_site_graph = G.subgraph(binding_site)
            
            isomorphisms = find_isomorphism_bindingsite(G, binding_site_graph)

            aligned_lig_coords = [ligand_pos]
            aligned_lig_graphs = list()
            iso_index = 1
            
            if visualize:
                ligand_graph = build_ca_graph(ligand_pos, cutoff=LIGAND_CUTOFF, node_type=colors)
                save_3d_graph_(ligand_graph, filename=f"{complex_graph['metadata']['sample_ID']}_ligand_graph.html")
                save_3d_graph_(binding_site_graph, filename=f"{complex_graph['metadata']['sample_ID']}_binding_site.html")
                save_3d_graph_(G, filename=f"{complex_graph['metadata']['sample_ID']}_protein.html")
            
            accepted_iso_list = []    
            
            for iso in isomorphisms:
                if check_identity_dict(iso):
                    accepted_iso_list.append(iso)
                    continue
                
            for iso in isomorphisms:
                if check_identity_dict(iso):
                    continue
                
                accept = True
                for iso_ref in accepted_iso_list:
                    source = list(iso_ref.keys())
                    target = list(iso.keys())
                
                    source_torch = torch.tensor(source)
                    target_torch = torch.tensor(target)
                    to_compare = source_torch != target_torch
                    if to_compare.sum() == 0:
                        accept = False
                        break
                
                    similarity_transform = corresponding_points_alignment(
                        residue_coords[source][to_compare].reshape(1, -1, 3), residue_coords[target][to_compare].reshape(1, -1, 3), weights=None, estimate_scale=False
                    )
                    aligned_bind_coords = apply_similarity_transform(residue_coords[source].reshape(1, -1, 3), *similarity_transform).reshape(-1, 3)
                    # double check the binding site
                    distance = (residue_coords[target] - aligned_bind_coords).norm(dim=-1).mean()
                    
                    if distance > 0.8:
                        accept = False
                        break
                    
                    
                if accept:
                    accepted_iso_list.append(iso)
                    
            for iso in accepted_iso_list:
                if check_identity_dict(iso):
                    continue
                source = list(iso.values())
                target = list(iso.keys())
                
                similarity_transform = corresponding_points_alignment(
                    residue_coords[source].reshape(1, -1, 3), residue_coords[target].reshape(1, -1, 3), weights=None, estimate_scale=False
                )
                aligned_bind_coords = apply_similarity_transform(residue_coords[source].reshape(1, -1, 3), *similarity_transform).reshape(-1, 3)
                # double check the binding site
                distance = (residue_coords[target] - aligned_bind_coords).norm(dim=-1).mean()
                distance_ori = (residue_coords[source] - residue_coords[target]).norm(dim=-1).mean()
                    
            
                aligned_lig_coord = apply_similarity_transform(ligand_pos.reshape(1, -1, 3), *similarity_transform).reshape(-1, 3)
                aligned_lig_coords.append(aligned_lig_coord)
                if visualize:
                    sample_id = complex_graph["metadata"]["sample_ID"]
                    aligned_lig_graph = build_ca_graph(aligned_lig_coord, cutoff=LIGAND_CUTOFF)
                    aligned_lig_graphs.append(aligned_lig_graph)
                    save_3d_graph(G, ligand_graph, aligned_lig_graphs[-1], binding_site, isomorphisms, filename=f"{sample_id}_{iso_index}_pl_graph.html")
                    save_3d_graph(G, ligand_graph, aligned_lig_graphs[-1], binding_site, isomorphisms, filename=f"{sample_id}_{iso_index}_pl_graph_binding_site.html", only_binding_site=True)
                    save_3d_graph(G, ligand_graph, aligned_lig_graphs[-1], binding_site, isomorphisms, filename=f"{sample_id}_{iso_index}_pl_graph_binding_site_om.html", only_binding_site=True, only_molecule=True)

                    iso_index += 1
                
            complex_graph["metadata"]["num_sim_binding_sites"] = len(aligned_lig_coords)
            complex_graph["features"]["aligned_lig_coords"] = torch.cat(aligned_lig_coords, dim=0)
            # except Exception as e:
            #     log.info(f"Error in generating all permutations: {e}")
            #     complex_graph["features"]["all_perms"] = torch.full((self.max_perm, num_nodes), -1, dtype=torch.long).t()
            #     complex_graph["metadata"]["num_perms"] = 0
            #     complex_graph["metadata"]["max_perm"] = self.max_perm
            
        complex_graph = align_apo_to_holo(complex_graph)
        return complex_graph

    def preprocessing(self):
        """Preprocesses the complexes for training."""
        log.info(
            f"Processing complexes from [{self.split_path}] and saving them to [{self.full_cache_path}]"
        )

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]
        log.info(f"Loading {len(complex_names_all)} complexes.")

        if self.esm_embeddings_path is not None:
            log.info("Loading ESM embeddings")
            chain_embeddings_dictlist = defaultdict(list)
            chain_indices_dictlist = defaultdict(list)
            for embedding_filepath in os.listdir(self.esm_embeddings_path):
                key = Path(embedding_filepath).stem
                key_name = key.split("_chain_")[0]
                if key_name in complex_names_all:
                    embedding = torch.load(
                        os.path.join(self.esm_embeddings_path, embedding_filepath)
                    )["representations"][33]
                    chain_embeddings_dictlist[key_name].append(embedding)
                    chain_indices_dictlist[key_name].append(int(key.split("_chain_")[1]))
            lm_embeddings_chains_all = []
            for name in complex_names_all:
                complex_chains_embeddings = chain_embeddings_dictlist[name]
                complex_chains_indices = chain_indices_dictlist[name]
                chain_reorder_idx = np.argsort(complex_chains_indices)
                reordered_chains = {
                    idx: complex_chains_embeddings[i] for idx, i in enumerate(chain_reorder_idx)
                }
                lm_embeddings_chains_all.append(reordered_chains)
        else:
            lm_embeddings_chains_all = [None] * len(complex_names_all)

        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(complex_names_all) // 1000 + 1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                continue
            complex_names = complex_names_all[1000 * i : 1000 * (i + 1)]
            lm_embeddings_chains = lm_embeddings_chains_all[1000 * i : 1000 * (i + 1)]
            complex_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(
                total=len(complex_names),
                desc=f"Loading complexes {i}/{len(complex_names_all)//1000+1}",
            ) as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(
                    self.get_complex,
                    zip(
                        complex_names,
                        lm_embeddings_chains,
                        [None] * len(complex_names),
                        [None] * len(complex_names),
                    ),
                ):
                    if t is not None:
                        complex_graphs.extend(t[0])
                        rdkit_ligands.extend(t[1])
                    pbar.update()
            if self.num_workers > 1:
                p.__exit__(None, None, None)

            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), "wb") as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), "wb") as f:
                pickle.dump((rdkit_ligands), f)

    def inference_preprocessing(self):
        """Preprocesses the complexes for inference."""
        ligands_list = []
        log.info("Reading molecules and generating local structures with RDKit")
        for ligand_description in tqdm(self.ligand_descriptions):
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path
            if mol is not None:
                mol = AddHs(mol)
                generate_conformer(mol)
                ligands_list.append(mol)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if not self.keep_local_structures:
                    mol.RemoveAllConformers()
                    mol = AddHs(mol)
                    generate_conformer(mol)
                ligands_list.append(mol)

        if self.esm_embeddings_path is not None:
            log.info("Reading language model embeddings.")
            lm_embeddings_chains_all = []
            if not os.path.exists(self.esm_embeddings_path):
                raise Exception("ESM embeddings path does not exist: ", self.esm_embeddings_path)
            for protein_path in self.protein_path_list:
                embeddings_paths = sorted(
                    glob.glob(
                        os.path.join(self.esm_embeddings_path, os.path.basename(protein_path))
                        + "*"
                    )
                )
                lm_embeddings_chains = []
                for embeddings_path in embeddings_paths:
                    lm_embeddings_chains.append(torch.load(embeddings_path)["representations"][33])
                lm_embeddings_chains_all.append(lm_embeddings_chains)
        else:
            lm_embeddings_chains_all = [None] * len(self.protein_path_list)

        log.info("Generating graphs for ligands and proteins")
        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(self.protein_path_list) // 1000 + 1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                continue
            protein_paths_chunk = self.protein_path_list[1000 * i : 1000 * (i + 1)]
            ligand_description_chunk = self.ligand_descriptions[1000 * i : 1000 * (i + 1)]
            ligands_chunk = ligands_list[1000 * i : 1000 * (i + 1)]
            lm_embeddings_chains = lm_embeddings_chains_all[1000 * i : 1000 * (i + 1)]
            complex_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(
                total=len(protein_paths_chunk),
                desc=f"Loading complexes {i}/{len(protein_paths_chunk)//1000+1}",
            ) as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(
                    self.get_complex,
                    zip(
                        protein_paths_chunk,
                        lm_embeddings_chains,
                        ligands_chunk,
                        ligand_description_chunk,
                    ),
                ):
                    if t is not None:
                        complex_graphs.extend(t[0])
                        rdkit_ligands.extend(t[1])
                    pbar.update()
            if self.num_workers > 1:
                p.__exit__(None, None, None)

            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), "wb") as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), "wb") as f:
                pickle.dump((rdkit_ligands), f)

    def check_all_complexes(self):
        """Checks if all complexes are already in the cache."""
        if os.path.exists(os.path.join(self.full_cache_path, "heterographs.pkl")):
            return True

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]
        for i in range(len(complex_names_all) // 1000 + 1):
            if not os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                return False
        return True

    def collect_all_complexes(self):
        """Collects all complexes from the cache."""
        log.info("Collecting all complexes from cache", self.full_cache_path)
        if os.path.exists(os.path.join(self.full_cache_path, "heterographs.pkl")):
            with open(os.path.join(self.full_cache_path, "heterographs.pkl"), "rb") as f:
                complex_graphs = pickle.load(f)  # nosec
            if self.require_ligand:
                with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), "rb") as f:
                    rdkit_ligands = pickle.load(f)  # nosec
            else:
                rdkit_ligands = None
            return complex_graphs, rdkit_ligands

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]
        complex_graphs_all = []
        for i in range(len(complex_names_all) // 1000 + 1):
            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), "rb") as f:
                log.info(f"Loading heterographs{i}.pkl")
                item = pickle.load(f)  # nosec
                complex_graphs_all.extend(item)

        rdkit_ligands_all = []
        for i in range(len(complex_names_all) // 1000 + 1):
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), "rb") as f:
                item = pickle.load(f)  # nosec
                rdkit_ligands_all.extend(item)

        return complex_graphs_all, rdkit_ligands_all

    def get_complex(self, par):
        """Returns a list of HeteroData objects and a list of RDKit molecules for a given
        complex."""
        name, lm_embedding_chains, ligand, _ = par
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)) and ligand is None:
            log.error(f"Data directory not found for {name}")
            return [], []

        try:
            lig = read_mol(self.pdbbind_dir, name, suffix=self.ligand_file, remove_hs=False)
            if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                log.error(
                    f"Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Skipping preprocessing for this example..."
                )
                return [], []

            if self.remove_hs:
                lig = RemoveHs(lig)

            lig_samples = [
                process_molecule(
                    lig,
                    ref_conf_xyz=np.array(lig.GetConformer().GetPositions()),
                    return_as_dict=True,
                )
            ]

        except Exception as e:
            log.error(f"Skipping {name} because of error: {e}")
            return [], []

        try:
            holo_protein_filepath = os.path.join(
                self.pdbbind_dir, name, f"{name}_{self.protein_file}.pdb"
            )
            holo_af_protein = pdb_filepath_to_protein(holo_protein_filepath)
            holo_protein_sample = process_protein(
                holo_af_protein,
                sample_name=f"{name}_",
            )
            complex_graph = merge_protein_and_ligands(
                lig_samples,
                holo_protein_sample,
                n_lig_patches=self.n_lig_patches,
            )

        except Exception as e:
            log.error(f"Skipping holo {name} because of error: {e}")
            return [], []

        if np.isnan(complex_graph["features"]["res_atom_positions"]).any():
            log.error(
                f"NaN in holo receptor pos for {name}. Skipping preprocessing for this example..."
            )
            return None

        try:
            if self.apo_protein_structure_dir is not None:
                apo_protein_filepath = os.path.join(
                    self.apo_protein_structure_dir, f"{name}_holo_aligned_esmfold_protein.pdb"
                )
                apo_af_protein = pdb_filepath_to_protein(apo_protein_filepath)
                apo_protein_sample = process_protein(
                    apo_af_protein,
                    sample_name=f"{name}_",
                    sequences_to_embeddings=lm_embedding_chains,
                )
                for key in complex_graph.keys():
                    for subkey, value in apo_protein_sample[key].items():
                        complex_graph[key]["apo_" + subkey] = value
                if not np.array_equal(
                    complex_graph["features"]["res_type"],
                    complex_graph["features"]["apo_res_type"],
                ):
                    log.error(
                        f"Residue type mismatch between holo protein and apo protein for {name}. Skipping preprocessing for this example..."
                    )
                    return None
                if np.isnan(complex_graph["features"]["apo_res_atom_positions"]).any():
                    log.error(
                        f"NaN in apo receptor pos for {name}. Skipping preprocessing for this example..."
                    )
                    return None

        except Exception as e:
            log.error(f"Skipping apo {name} because of error: {e}")
            return [], []

        if (
            self.min_protein_length is not None
            and complex_graph["metadata"]["num_a"] < self.min_protein_length
            and not self.is_test_dataset
        ):
            log.info(f"Skipping {name} because of its length {complex_graph['metadata']['num_a']}")
            return [], []
        if (
            self.max_protein_length is not None
            and complex_graph["metadata"]["num_a"] > self.max_protein_length
            and not self.is_test_dataset
        ):
            log.info(f"Skipping {name} because of its length {complex_graph['metadata']['num_a']}")
            return [], []

        complex_graph["metadata"]["sample_ID"] = name
        return [complex_graph], [lig]


def read_mol(pdbbind_dir, name, suffix="ligand", remove_hs=False):
    """Reads a ligand from the given directory and returns it as an RDKit molecule."""
    lig = read_molecule(
        os.path.join(pdbbind_dir, name, f"{name}_{suffix}.mol2"),
        remove_hs=remove_hs,
        sanitize=True,
    )
    if lig is None:  # read sdf file if mol2 file cannot be sanitized
        log.info(
            "Reading the .mol2 file failed. We found a .sdf file instead and are trying to use that. Be aware that the .sdf files from PDBBind 2020 are missing chirality tags, although we will do our best to impute such information automatically using RDKit. Reference: https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind."
        )
        lig = read_molecule(
            os.path.join(pdbbind_dir, name, f"{name}_{suffix}.sdf"),
            remove_hs=remove_hs,
            sanitize=True,
        )
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(lig)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    """Reads all ligands from the given directory and returns them as a list of RDKit molecules."""
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".mol2") and "rdkit" not in file:
            lig = read_molecule(
                os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True
            )
            if lig is None and os.path.exists(
                os.path.join(pdbbind_dir, name, file[:-4] + ".sdf")
            ):  # read sdf file if mol2 file cannot be sanitized
                log.info(
                    "Using the .mol2 file failed. We found a .sdf file instead and are trying to use that. Be aware that the .sdf files from PDBBind 2020 are missing chirality tags, although we will do our best to impute such information automatically using RDKit. Reference: https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind."
                )
                lig = read_molecule(
                    os.path.join(pdbbind_dir, name, file[:-4] + ".sdf"),
                    remove_hs=remove_hs,
                    sanitize=True,
                )
                Chem.rdmolops.AssignAtomChiralTagsFromStructure(lig)
            if lig is not None:
                ligs.append(lig)
    return ligs
