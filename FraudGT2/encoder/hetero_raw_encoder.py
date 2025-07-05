import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from config.config import cfg
#from fraudGT.graphgym.register import register_node_encoder, register_edge_encoder


# Deleted the class called GeneralLayer

class HeteroRawNodeEncoder(torch.nn.Module):
    """
    The raw feature node encoder

    Args:
        emb_dim (int): Output embedding dimension
    """
    def __init__(self, dim_emb, dataset):
        super().__init__()
        self.dim_in = cfg.share.dim_in
        if cfg.train.add_ego_id:
            self.dim_in = {
                node_type: self.dim_in[node_type] + 1 for node_type in self.dim_in
            }
        self.metadata = dataset[0].metadata()
        data = dataset[0]
        # pecfg           = cfg.posenc_Hetero_Raw
        # self.layers     = pecfg.layers
        # self.dropout    = pecfg.dropout
        # self.act        = pecfg.act
        
        if not isinstance(dim_emb, dict):
            dim_emb = {node_type: dim_emb for node_type in data.node_types}
        
        self.linear = nn.ModuleDict()
        for node_type in self.metadata[0]:
            if hasattr(data[node_type], 'x'):
                self.linear[node_type] = nn.Linear(
                    self.dim_in[node_type], dim_emb[node_type])
            # self.linear[node_type] = GeneralMultiLayer('linear', 
            #                  self.pre_layers, self.dim_in[node_type], dim_emb[node_type],
            #                  dim_inner=dim_emb[node_type], final_act=True,
            #                  has_bn=self.batch_norm, has_ln=self.layer_norm,
            #                  dropout=self.dropout, act=self.act)
            else:
                self.dim_in[node_type] = dim_emb[node_type]

        self.encoder = nn.ModuleDict(
            {
                node_type: nn.Embedding(data[node_type].num_nodes, dim_emb[node_type])
                for node_type in data.node_types
                if not hasattr(data[node_type], 'x')
            }
        )
        
    def forward(self, batch):
        if isinstance(batch, HeteroData):
            # Only changing the x itself can make sure the to_homogeneous() function works well later
            for node_type in batch.node_types:
                if hasattr(batch[node_type], 'x'):
                    batch[node_type].x = self.linear[node_type](batch[node_type].x)
                else:
                    if cfg.train.sampler == 'full_batch':
                        batch[node_type].x = self.encoder[node_type].weight.data #(torch.arange(batch[node_type].num_nodes))
                    else:
                        batch[node_type].x = self.encoder[node_type](batch[node_type].n_id)
        else:
            x = batch.x
            batch.x = list(self.linear.values())[0](x)

        return batch


class HeteroRawEdgeEncoder(torch.nn.Module):
    """
    The raw feature edge encoder

    Args:
        emb_dim (int): Output embedding dimension
    """
    def __init__(self, dim_emb, dataset):
        super().__init__()
        self.dim_in = cfg.share.dim_in
        self.metadata = dataset[0].metadata()
        
        if not isinstance(dim_emb, dict):
            dim_emb = {edge_type: dim_emb for edge_type in self.dim_in}
        
        self.linear = nn.ModuleDict()
        for edge_type in self.metadata[1]:
            edge_type = '__'.join(edge_type)
            self.linear[edge_type] = nn.Linear(
                self.dim_in[edge_type], dim_emb[edge_type])
    
    def forward(self, batch):
        # print(batch)
        # print(batch[('node', 'to', 'node')].e_id)
        # print(batch[('node', 'to', 'node')].input_id)
        # print(torch.isin(batch[('node', 'to', 'node')].e_id, batch[('node', 'to', 'node')].input_id).sum())
        # print(batch[('node', 'to', 'node')].edge_index)
        # print(batch[('node', 'to', 'node')].edge_label_index)

        if isinstance(batch, HeteroData):
            # Only changing the x itself can make sure the to_homogeneous() function works well later
            for edge_type, edge_attr in batch.collect("edge_attr").items():
                batch[edge_type].edge_attr = self.linear['__'.join(edge_type)](edge_attr) 
        else:
            edge_attr = batch.edge_attr
            batch.edge_attr = list(self.linear.values())[0](edge_attr)

        return batch