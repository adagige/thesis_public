from typing import Callable

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b,
                                      TUDataset)
from torch_geometric.loader import (ClusterLoader, DataLoader,
                                    GraphSAINTEdgeSampler,
                                    GraphSAINTNodeSampler,
                                    GraphSAINTRandomWalkSampler,
                                    NeighborSampler, RandomNodeSampler)
from torch_geometric.utils import (index_to_mask, negative_sampling,
                                   to_undirected)

import fraudGT.graphgym.register as register
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.transform import create_link_label, neg_sampling_transform
from fraudGT.datasets.temporal_dataset import TemporalDataset
from yacs.config import CfgNode as CN


def planetoid_dataset(name: str) -> Callable:
    return lambda root: Planetoid(root, name)


register.register_dataset('Cora', planetoid_dataset('Cora'))
register.register_dataset('CiteSeer', planetoid_dataset('CiteSeer'))
register.register_dataset('PubMed', planetoid_dataset('PubMed'))
register.register_dataset('PPI', PPI)


def load_pyg(name, dataset_dir):
    """
    Load PyG dataset objects. (More PyG datasets will be supported)

    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset = TUDataset(dataset_dir, name, transform=T.Constant())
        else:
            dataset = TUDataset(dataset_dir, name[3:])
    elif name == 'Karate':
        dataset = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset = Coauthor(dataset_dir, name='CS')
        else:
            dataset = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset = Amazon(dataset_dir, name='Computers')
        else:
            dataset = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))

    return dataset


def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


def load_ogb(name, dataset_dir):
    r"""

    Load OGB dataset objects.


    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.linkproppred import PygLinkPropPredDataset
    from ogb.nodeproppred import PygNodePropPredDataset

    if name[:4] == 'ogbn':
        dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for i, key in enumerate(splits.keys()):
            mask = index_to_mask(splits[key], size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_names[i], mask, len(mask))
        edge_index = to_undirected(dataset.data.edge_index)
        set_dataset_attr(dataset, 'edge_index', edge_index,
                         edge_index.shape[1])

    elif name[:4] == 'ogbg':
        dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        for i, key in enumerate(splits.keys()):
            id = splits[key]
            set_dataset_attr(dataset, split_names[i], id, len(id))

    elif name[:4] == "ogbl":
        dataset = PygLinkPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_edge_split()
        id = splits['train']['edge'].T
        if cfg.dataset.resample_negative:
            set_dataset_attr(dataset, 'train_pos_edge_index', id, id.shape[1])
            dataset.transform = neg_sampling_transform
        else:
            id_neg = negative_sampling(edge_index=id,
                                       num_nodes=dataset.data.num_nodes,
                                       num_neg_samples=id.shape[1])
            id_all = torch.cat([id, id_neg], dim=-1)
            label = create_link_label(id, id_neg)
            set_dataset_attr(dataset, 'train_edge_index', id_all,
                             id_all.shape[1])
            set_dataset_attr(dataset, 'train_edge_label', label, len(label))

        id, id_neg = splits['valid']['edge'].T, splits['valid']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'val_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'val_edge_label', label, len(label))

        id, id_neg = splits['test']['edge'].T, splits['test']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'test_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'test_edge_label', label, len(label))

    else:
        raise ValueError('OGB dataset: {} non-exist')
    return dataset


def load_dataset():
    r"""

    Load dataset objects.

    Returns: PyG dataset object

    """
    format = cfg.dataset.format
    print('Format is', format)
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    print('Loader keys',  register.loader_dict.keys())
    for func in register.loader_dict.values():
        dataset = func(format, name, dataset_dir)
        
        if dataset is not None:
            return dataset
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    # Load from OGB formatted data
    elif format == 'OGB':
        dataset = load_ogb(name.replace('_', '-'), dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(format))
    return dataset


def set_dataset_info(dataset):
    r"""
    Set global dataset information

    Args:
        dataset: PyG dataset object

    """

    # get dim_in and dim_out
    try:
        if isinstance(dataset.data, HeteroData): # Hetero graph
            cfg.share.dim_in = CN()
            task = cfg.dataset.task_entity
            try:
                if hasattr(dataset.data, "x_dict"):
                    for node_type in dataset.data.node_types:
                        if node_type in dataset.data.x_dict:
                            cfg.share.dim_in[node_type] = dataset.data.x_dict[node_type].shape[1]
                        else:
                            cfg.share.dim_in[node_type] = None
            except:
                pass # No x_dict
            try:
                if hasattr(dataset.data, "edge_attr_dict"):
                    for edge_type in dataset.data.edge_types:
                        if edge_type in dataset.data.edge_attr_dict:
                            # Key doesn't support tuple
                            cfg.share.dim_in["__".join(edge_type)] = dataset.data.edge_attr_dict[edge_type].shape[1]
                        else:
                            cfg.share.dim_in["__".join(edge_type)] = None
            except:
                pass # No edge_attr_dict
        else:
            cfg.share.dim_in = dataset.data.x.shape[1]
    except Exception:
        cfg.share.dim_in = 1
        
    try:
        if cfg.dataset.task_type == 'classification':
            if isinstance(dataset.data, HeteroData): # Hetero graph
                task = cfg.dataset.task_entity
                if hasattr(dataset.data[task], 'y'):
                    y = dataset.data[task].y
                elif hasattr(dataset.data[task], 'edge_label'):
                    y = dataset.data[task].edge_label
            else:
                if hasattr(dataset.data, 'y'):
                    y = dataset.data.y
                elif hasattr(dataset.data, 'edge_label'):
                    y = dataset.data.edge_label

            if y.numel() == y.size(0) and not torch.is_floating_point(y):
                cfg.share.dim_out = int(y.max()) + 1
            elif y.numel() == y.size(0) and torch.is_floating_point(y):
                cfg.share.dim_out = torch.unique(y).numel()
            else:
                cfg.share.dim_out = y.size[-1]
        else:
            if isinstance(dataset.data, HeteroData):
                task = cfg.dataset.task_entity
                if hasattr(dataset.data[task], 'y'):
                    y = dataset.data[task].y
                elif hasattr(dataset.data[task], 'edge_label'):
                    y = dataset.data[task].edge_label
            else:
                if hasattr(dataset.data, 'y'):
                    y = dataset.data.y
                elif hasattr(dataset.data, 'edge_label'):
                    y = dataset.data.edge_label

            cfg.share.dim_out = y.shape[-1]
    except Exception:
        cfg.share.dim_out = 1

    # count number of dataset splits
    cfg.share.num_splits = 1
    for key in dataset.data.keys():
        if 'val' in key:
            cfg.share.num_splits += 1
            break
    for key in dataset.data.keys():
        if 'test' in key:
            cfg.share.num_splits += 1
            break

    if isinstance(dataset, TemporalDataset):
        cfg.share.num_splits = len(dataset)


def create_dataset():
    r"""
    Create dataset object

    Returns: PyG dataset object

    """
    print('Create dataset function')
    dataset = load_dataset()
    set_dataset_info(dataset)

    return dataset


def get_loader(dataset, sampler, batch_size, shuffle=True, split='train'):
    # Try to use customized graph sampler
    func = register.sampler_dict.get(sampler, None)
    if func is not None:
        return func(dataset, batch_size=batch_size, shuffle=shuffle, split=split)
    
    if sampler == "full_batch" or len(dataset) > 1:
        loader_train = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "neighbor":
        loader_train = NeighborSampler(
            dataset[0],
            sizes=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True)
    elif sampler == "random_node":
        loader_train = RandomNodeSampler(dataset[0],
                                         num_parts=cfg.train.train_parts,
                                         shuffle=shuffle,
                                         num_workers=cfg.num_workers,
                                         pin_memory=True)
    elif sampler == "saint_rw":
        loader_train = \
            GraphSAINTRandomWalkSampler(dataset[0],
                                        batch_size=batch_size,
                                        walk_length=cfg.train.walk_length,
                                        num_steps=cfg.train.iter_per_epoch,
                                        sample_coverage=0,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True)
    elif sampler == "saint_node":
        loader_train = \
            GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "saint_edge":
        loader_train = \
            GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "cluster":
        loader_train = \
            ClusterLoader(dataset[0],
                          num_parts=cfg.train.train_parts,
                          save_dir="{}/{}".format(cfg.dataset.dir,
                                                  cfg.dataset.name.replace(
                                                      "-", "_")),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers,
                          pin_memory=True)
    else:
        raise NotImplementedError("%s sampler is not implemented!" % sampler)
    return loader_train


def create_loader(dataset = None, shuffle = True, returnDataset = False):
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    if dataset is None:
        print(dataset,'creating dataset bla')
        dataset = create_dataset()
        
    print('Load dataset')

    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                        shuffle=True)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset,
                        cfg.train.sampler,
                        cfg.train.batch_size,
                        shuffle=True,
                        split='train')
        ]
    print('Create train loader')

    # val and test loaders
    split_names = ['val', 'test']
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                           shuffle=shuffle))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                get_loader(dataset,
                            cfg.val.sampler,
                            cfg.train.batch_size,
                            shuffle=shuffle,
                            split=split_names[i]))
            
    print('Create val/test loader')

    if returnDataset:
        return loaders, dataset
    else:
        return loaders
