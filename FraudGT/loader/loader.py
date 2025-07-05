
from config.config import cfg
from torch_geometric.data import HeteroData
from yacs.config import CfgNode as CN
import torch
from datasets.temporal_dataset import TemporalDataset

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



def load_dataset():
    r"""

    Load dataset objects.

    Returns: PyG dataset object

    """
    format = cfg.dataset.format
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
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






def create_dataset():
    r"""
    Create dataset object

    Returns: PyG dataset object

    """
    dataset = load_dataset()
    set_dataset_info(dataset)

    return dataset


def create_loader(dataset = None, shuffle = True, returnDataset = False):
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    if dataset is None:
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