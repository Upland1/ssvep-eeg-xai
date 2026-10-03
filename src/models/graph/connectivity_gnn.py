"""Dense edge-gated GNN for SSVEP frequency classification.

The model uses node features, learned channel embeddings, message passing,
optional edge gates, and mean/max pooling. Graph-level features such as FBCCA
scores are added after pooling.
"""

import torch
import torch.nn as nn


class EdgeGatedLayer(nn.Module):

  def __init__(self, dim: int, edge_dim: int, dropout: float):
    super().__init__()
    self.w_self = nn.Linear(dim, dim, bias=False)
    self.w_msg = nn.Linear(dim, dim, bias=False)
    self.edge_gate = nn.Sequential(nn.Linear(edge_dim, dim), nn.Sigmoid()) if edge_dim > 0 else None
    self.norm = nn.LayerNorm(dim)
    self.drop = nn.Dropout(dropout)
    self.act = nn.ELU()

  def forward(self, h: torch.Tensor, e: torch.Tensor | None) -> torch.Tensor:
    b, n, _ = h.shape
    msg = self.w_msg(h)
    # Average messages from other nodes.
    mask = (1.0 - torch.eye(n, device=h.device, dtype=h.dtype)) / max(n - 1, 1)   # (N, N)
    if e is None:
      agg = torch.einsum("ij,bjd->bid", mask, msg)
    else:
      gate = self.edge_gate(e)
      agg = (gate * mask[None, :, :, None] * msg.unsqueeze(1)).sum(dim=2)
    return self.norm(h + self.drop(self.act(self.w_self(h) + agg)))


class ConnectivityGNN(nn.Module):

  def __init__(
      self,
      n_nodes: int,
      n_node_feats: int,
      n_edge_feats: int = 0,
      n_graph_feats: int = 0,
      n_classes: int = 5,
      hidden: int = 32,
      n_layers: int = 2,
      dropout: float = 0.3,
  ):
    super().__init__()
    self.n_nodes = n_nodes
    self.node_in = nn.Sequential(nn.Linear(n_node_feats, hidden), nn.ELU())
    self.channel_embed = nn.Parameter(torch.zeros(1, n_nodes, hidden))
    nn.init.normal_(self.channel_embed, std=0.1)
    self.layers = nn.ModuleList(
        [EdgeGatedLayer(hidden, n_edge_feats, dropout) for _ in range(n_layers)]
    )
    # Keep graph-level features intact; regularise only the learned read-out.
    self.readout_drop = nn.Dropout(dropout)
    self.head = nn.Linear(2 * hidden + n_graph_feats, n_classes)

  def forward(
      self,
      nodes: torch.Tensor,
      edges: torch.Tensor | None = None,
      graph: torch.Tensor | None = None,
  ) -> torch.Tensor:
    h = self.node_in(nodes) + self.channel_embed
    for layer in self.layers:
      h = layer(h, edges)
    readout = self.readout_drop(torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1))
    if graph is not None:
      readout = torch.cat([readout, graph], dim=-1)
    return self.head(readout)