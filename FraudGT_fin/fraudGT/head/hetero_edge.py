import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index

from fraudGT.graphgym.register import register_head
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.layer import MLP


@register_head('hetero_edge')
class HeteroGNNEdgeHead(nn.Module):
    '''Head of Hetero GNN, edge prediction'''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        # self.train_edge_inds = mask_to_index(data[cfg.dataset.task_entity].train_edge_mask).to(cfg.device)
        # self.val_edge_inds = mask_to_index(data[cfg.dataset.task_entity].val_edge_mask).to(cfg.device)
        # self.test_edge_inds = mask_to_index(data[cfg.dataset.task_entity].test_edge_mask).to(cfg.device)
        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        print('dim_in is this:',dim_in)

        self.layer_post_mp = MLP(dim_in * 3, dim_out, 
                                 num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
                                 bias=True)
        # requires parameter
        # self.decode_module = lambda v1, v2: \
        #     self.layer_post_mp(torch.cat((v1, v2), dim=-1))

    def _apply_index(self, batch,return_edges=False, only_test=False):
        task = cfg.dataset.task_entity
        # There could be multi-edge between node pair, using edge id is the safest way
        # mask = torch.isin(getattr(self, f'{batch.split}_edge_inds')[batch[task].e_id], 
        #                   getattr(self, f'{batch.split}_inds')[batch[task].input_id])
        # print("Total edges in batch:", batch[task].e_id.shape)
        # print("Valid indices in split:", getattr(self, f'{batch.split}_inds').shape)
        # print('batch[task].e_id', batch[task].e_id.size(), flush=True)
        # print("Batch input_id shape:", batch[task].input_id.shape,flush=True)
        # print("Edges that match input_id:", torch.isin(batch[task].e_id, batch[task].input_id).sum())

        

        mask = torch.isin(batch[task].e_id, 
                          getattr(self, f'{batch.split}_inds')[batch[task].input_id])
        #NEW MASK BY ADA
      #  mask = torch.isin(batch[task].e_id, getattr(self, f'{batch.split}_inds'))
        # print('TASK:\n\n:', task,flush=True)
        # print("Number of edges after mask:", mask.sum())
        task = cfg.dataset.task_entity
        edge_index = batch[task].edge_index



        # A concatentation of source/target node embedding + edge attribute
        if return_edges:
          if only_test:
            return torch.cat((batch[task[0]].x[edge_index[0, mask]], 
                              batch[task[2]].x[edge_index[1, mask]], 
                              batch[task].edge_attr[mask]), dim=-1), \
                  batch[task].y[mask],batch[task].edge_id[mask]

          elif only_test==False:
            return torch.cat((batch[task[0]].x[edge_index[0,:]], 
                              batch[task[2]].x[edge_index[1,:]], 
                              batch[task].edge_attr), dim=-1), \
                  batch[task].y,batch[task].edge_id

        else:
          return torch.cat((batch[task[0]].x[edge_index[0, mask]], 
                            batch[task[2]].x[edge_index[1, mask]], 
                            batch[task].edge_attr[mask]), dim=-1), \
                batch[task].y[mask],batch[task].edge_id[mask]
    

    def forward(self, batch, return_edges=False, only_test=False):
        # TODO: add homogeneous graph support
        # batch.x_dict[cfg.dataset.task_entity] = self.layer_post_mp(batch.x_dict[cfg.dataset.task_entity])
        # pred, label = self._apply_index(batch)
    
        # if cfg.model.edge_decoding != 'concat':
        #     batch = self.layer_post_mp(batch)
      #  print('Batch after gt before head looks like: \n\n', batch, flush=True)
        pred, label,edge_id = self._apply_index(batch,return_edges, only_test)
      #  print('Prediction in head looks like: \n\n',pred.size(), flush=True)
       # print('Edge id looks like: ', edge_id.size(), flush=True)
        # nodes_first = pred[0]
        # nodes_second = pred[1]
        # pred = self.decode_module(nodes_first, nodes_second)
        pred = self.layer_post_mp(pred)
      #  print('Batch after head looks like: \n\n', batch, flush=True)

        if return_edges:
            return pred, label, edge_id
        else:
            return pred, label
    
        # if not self.training:  # Compute extra stats when in evaluation mode.
        #     stats = self.compute_mrr(batch)
        #     return pred, label, stats
        # else:
        #     return pred, label
