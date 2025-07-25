import torch

from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.gnn import GNN
from fraudGT.graphgym.register import network_dict, register_network

register_network('gnn', GNN)


def create_model(to_device=True, dim_in=None, dim_out=None, dataset=None):
    r"""
    Create model for graph machine learning

    Args:
        to_device (string): The devide that the model will be transferred to
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    print('dimensions below\n \n')
    print(dim_in)
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    print(dim_in)
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out, dataset=dataset)
    print(network_dict.keys())
    print(network_dict.values())
    if to_device:
        model.to(torch.device(cfg.device))
    return model
