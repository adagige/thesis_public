import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from datasets.aml_dataset import AMLDataset
from sampler.sampler_custom import get_LinkNeighborLoader, AddEgoIdsForLinkNeighbor
from networks.gt_model import GTModel
from config.config import cfg
from loader.loader import set_dataset_info
from train.custom_train_cpu import custom_train as train
from logger import create_logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from typing import Tuple

#optimizer 
from torch.optim import AdamW
from torch_geometric.loader import LinkNeighborLoader 




dataset_dir = 'Documents/GitHub/thesis/fraudGT/aml_datasets'
name = 'Small-HI'
batch_size = 2048 #original 2048 
shuffle = True #Always set to true from the get_loader function


if __name__ == '__main__':
    dataset = AMLDataset(root=dataset_dir, name=name, reverse_mp=True, #cfg.dataset.reverse_mp,
                            add_ports= True#cfg.dataset.add_ports
                            )

    set_dataset_info(dataset)
    
   # print('dataset is here\n',dataset.data_dict['train'])

    # print(dataset['val'])
#     split = 'train'
#     task = ('node', 'to', 'node')
#     data = dataset.data_dict[split]
#   #  print(data[task])
#     mask = data[task].split_mask
#     #print(mask)
    
#     edge_label_index = data[task].edge_index[:, mask]
#     edge_label = data[task].y[mask]
#     #print('data[node, to, node].edge_label_index\n\n',data['node', 'to', 'node'].edge_label_index)

#   #  print(data['node', 'to', 'node']._mapping.keys())


#     train_loader = LinkNeighborLoader(data=data,
#                 num_neighbors= [10, 10], # cfg.train.neighbor_sizes,
#                 # neg_sampling_ratio=0.0,
#                 edge_label_index=(task, edge_label_index),
#                 edge_label=edge_label,
#                 batch_size=batch_size,
#                 num_workers= 2, #cfg.num_workers,
#                 shuffle=shuffle,
#                 transform=AddEgoIdsForLinkNeighbor())
    
    # for batch in train_loader:
    #     print("Batch Content:\n", batch)
    #     print(batch['node'].x)
       

    #Get training sampler 
    train_loader = get_LinkNeighborLoader(dataset.data_dict, batch_size=batch_size, shuffle=shuffle, split='train')

    val_loader = get_LinkNeighborLoader(dataset.data_dict, batch_size=batch_size, shuffle=shuffle, split='val')

    test_loader = get_LinkNeighborLoader(dataset.data_dict, batch_size=batch_size, shuffle=shuffle, split='test')

  #  train_loader, val_loader, test_loader  = loaders[0], loaders[1], loaders[2]

    
    # for batch in train_loader:
    #     print(batch)

    # iterator = iter(loaders[0])
    # batch = next(iterator, None)
    # print(batch)
    # sampled_data = next(iter(loaders[0]))
    # print(sampled_data)
    # for batch in next(iterator, None):
    #     print(batch)

    loggers = create_logger()

    #Initialize the model:
    model = GTModel(dim_in=cfg.share.dim_in, dim_out=1, dataset=dataset) #Virker som om dim_in er sat til 1 i den store config fil  (out put dim = 1 fordi classification task)

    # print(model)
    params = list(model.named_parameters())
    if isinstance(params[0], Tuple):
        params = filter(lambda p: p[1].requires_grad, params)
    else:
        params = filter(lambda p: p.requires_grad, params)

    weight_decay = 0.0

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]


    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=1e-5, lr=0.001)

    # #Scheduler dropped for now 
    # scheduler = None

    # model.to(cfg.device)
#print(loaders.split)

   # train(loggers, loaders, model, optimizer, scheduler)

    # Training loop
    num_epochs =  cfg.optim.max_epoch
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch.split = 'train'
            batch = batch.to(cfg.device)
            optimizer.zero_grad()
            pred, true = model(batch)#['node', 'to', 'node'].y
            # default manipulation for pred and true from loader.py
            # can be skipped if special loss computation is needed
            
            pred = pred.squeeze(-1) if pred.ndim > 1 else pred
            true = true.squeeze(-1) if true.ndim > 1 else true
        #    print(pred)
           # pred = torch.argmax(pred, dim=1)
           # pred = F.log_softmax(pred, dim=-1)
            
            weight = torch.tensor(cfg.model.loss_fun_weight, device=torch.device(cfg.device))

            loss = F.binary_cross_entropy_with_logits(pred.float(), true.float(), weight=weight[true])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.8f}')

    print('Training complete!')

    torch.save(model.state_dict(), r'C:\Users\adagi\Documents\GitHub\thesis\FraudGT\models\model.pth')
    print('Model saved successfully.')
    # if to_device:
    #   model.to(torch.device(cfg.device))
    # return model
