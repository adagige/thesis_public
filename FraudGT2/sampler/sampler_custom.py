
import torch
from typing import Union
from torch_geometric.data import (Data, HeteroData)
from torch_geometric.loader import LinkNeighborLoader 
from torch_geometric.transforms import BaseTransform

class LoaderWrapper:
    def __init__(self, dataloader, n_step=-1, split='train'):
        self.step = n_step if n_step > 0 else len(dataloader)
        self.idx = 0
        self.loader = dataloader
        self.split = split
        self.iter_loader = iter(dataloader)
    
    def __iter__(self):
        return self

    def __len__(self):
        if self.step > 0:
            return self.step
        else:
            return len(self.loader)

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.step or self.idx == len(self.loader):
            self.idx = 0
            if self.split in ['val', 'test']:
                # Make sure we are always using the same set of data for evaluation
                self.iter_loader = iter(self.loader)
            raise StopIteration
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)
    
    def set_step(self, n_step):
        self.step = n_step

# def convert_batch(batch):
#     for node_type, x in batch.x_dict.items():
#         batch[node_type].x = x.to(torch.float) 
#     for node_type, y in batch.y_dict.items():
#         batch[node_type].y = y.to(torch.long) 
#     return batch
class AddEgoIdsForLinkNeighbor(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        


#@register_sampler('link_neighbor')
def get_LinkNeighborLoader(dataset, batch_size, shuffle=True, split='train'):
    task = ('node', 'to', 'node') #cfg.dataset.task_entity
    data = dataset[split]
    mask = data[task].split_mask
    edge_label_index = data[task].edge_index[:, mask]
    edge_label = data[task].y[mask]
    loader_train = \
        LoaderWrapper( \
            LinkNeighborLoader(
                data=data,
                num_neighbors= [50, 50], # cfg.train.neighbor_sizes,
                # neg_sampling_ratio=0.0,
                edge_label_index=(task, edge_label_index),
                edge_label=edge_label,
                batch_size=batch_size,
                num_workers= 2, #cfg.num_workers,
                shuffle=shuffle,
                transform=AddEgoIdsForLinkNeighbor() #if cfg.train.add_ego_id else None
            ),
           10, # getattr(cfg, 'val' if split == 'test' else split).iter_per_epoch, #iterations per epoch, just set low now
            split
        )

    return loader_train