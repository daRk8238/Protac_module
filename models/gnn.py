"""
Graph Neural Network for Binding Affinity Prediction
------------------------------------------------------
Predicts pChEMBL value (binding affinity) from molecular graph.

Architecture: GCN/GAT encoder → global pooling → MLP regressor
Input:  Molecular graph (atoms as nodes, bonds as edges)
Output: Predicted pChEMBL value (float)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool


class MoleculeGNN(nn.Module):
    """
    Graph Neural Network for molecular property prediction.

    Three GCN layers encode atom neighborhoods, then global pooling
    aggregates to a fixed-size graph embedding, then an MLP regresses
    to pChEMBL value.
    """

    def __init__(
        self,
        node_features: int = 9,      # atom feature vector size (see featurizer)
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_gat: bool = False,        # swap GCN for GAT (attention-based)
        gat_heads: int = 4,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat

        # ── Graph Convolution Layers ──────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_dim = node_features
        for i in range(num_layers):
            if use_gat:
                out_dim = hidden_dim // gat_heads
                conv = GATConv(in_dim, out_dim, heads=gat_heads, dropout=dropout)
                in_dim = hidden_dim
            else:
                conv = GCNConv(in_dim, hidden_dim)
                in_dim = hidden_dim

            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # ── MLP Regressor ─────────────────────────────────────────────────────
        # global_mean_pool + global_max_pool concatenated → 2 * hidden_dim
        mlp_input_dim = 2 * hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),          # single output: predicted pChEMBL
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ── Message Passing ───────────────────────────────────────────────────
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:   # no dropout on last layer
                x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Global Pooling (mean + max concatenated) ──────────────────────────
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)

        # ── Regression ────────────────────────────────────────────────────────
        out = self.mlp(x_graph)
        return out.squeeze(1)


class AffinityPredictor(nn.Module):
    """
    Wrapper around MoleculeGNN with fingerprint fusion.
    Concatenates GNN graph embedding with Morgan fingerprint
    for improved accuracy on small datasets.
    """

    def __init__(
        self,
        node_features: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        fp_dim: int = 2048,           # Morgan fingerprint size
    ):
        super().__init__()

        # GNN encoder
        self.gnn = MoleculeGNN(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Fingerprint encoder (compress 2048 → 256)
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Final fusion MLP
        gnn_out_dim = 2 * hidden_dim   # mean + max pooling
        fusion_dim = gnn_out_dim + 256

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data, fingerprints=None):
        # GNN path
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.gnn.convs, self.gnn.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        gnn_emb = torch.cat([x_mean, x_max], dim=1)

        if fingerprints is not None:
            # Fusion path
            fp_emb = self.fp_encoder(fingerprints)
            combined = torch.cat([gnn_emb, fp_emb], dim=1)
            out = self.fusion(combined)
        else:
            # GNN only fallback
            out = self.gnn.mlp(gnn_emb)

        return out.squeeze(1)
