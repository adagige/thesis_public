import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

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
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import numpy as np

from typing import Tuple


dataset_dir = 'Documents/GitHub/thesis/fraudGT/aml_datasets'
name = 'Small-HI'
batch_size = 2048 #original 2048 
shuffle = True #Always set to true from the get_loader function
cfg.device = 'cpu'

if __name__ == '__main__':
    dataset = AMLDataset(root=dataset_dir, name=name, reverse_mp=True, #cfg.dataset.reverse_mp,
                            add_ports= True#cfg.dataset.add_ports
                            )

    set_dataset_info(dataset)



    loaders = [get_LinkNeighborLoader(dataset.data_dict, batch_size=batch_size, shuffle=shuffle, split='train')]

    loaders.append(get_LinkNeighborLoader(dataset.data_dict, batch_size=batch_size, shuffle=shuffle, split='val'))

    loaders.append(get_LinkNeighborLoader(dataset.data_dict, batch_size=batch_size, shuffle=shuffle, split='test'))

    train_loader, val_loader, test_loader  = loaders[0], loaders[1], loaders[2]
    def evaluate(model, data_loader, split):
        model.eval()
        total_loss = 0
        all_preds = np.array([])
        all_true = np.array([])
        with torch.no_grad():
            for batch in data_loader:
                batch.split = split
                batch = batch.to(cfg.device)
                pred, true = model(batch)#['node', 'to', 'node'].y
                # default manipulation for pred and true from loader.py
                # can be skipped if special loss computation is needed
                pred = pred.squeeze(-1) if pred.ndim > 1 else pred
                true = true.squeeze(-1) if true.ndim > 1 else true

                weight = torch.tensor(cfg.model.loss_fun_weight, device=torch.device(cfg.device))
                loss = F.binary_cross_entropy_with_logits(pred.float(), true.float(), weight=weight[true])
                total_loss += loss.item()
                pred = torch.round(torch.sigmoid(pred)) #Get the binary prediction to be used in f1 score
                
                all_preds = np.append(all_preds, pred.cpu().numpy())
                all_true = np.append(all_true, true.cpu().numpy())
    
        f1 = f1_score(all_true, all_preds, average='weighted')
        avg_loss = total_loss / len(val_loader)
        return avg_loss, f1, confusion_matrix(all_true, all_preds)

    model = GTModel(dim_in=cfg.share.dim_in, dim_out=1, dataset=dataset)
    model.load_state_dict(torch.load(r'C:\Users\adagi\Documents\GitHub\thesis\FraudGT\models\model_AML_Small_Hi.pth'))
    model.to(cfg.device)
    print('Model loaded successfully.')

    train_loss, train_f1,train_cf = evaluate(model, train_loader, 'train')
    val_loss, val_f1, val_cf = evaluate(model, val_loader, 'val')
    test_loss, test_f1, test_cf = evaluate(model, test_loader, 'test')

    print('Train loss',train_loss, 'Train F1-Score',train_f1, 'Confusion matrix:', train_cf)
    print('Validataion loss',val_loss,'Validation F1-Score',val_f1,'Confusion matrix:', val_cf)
    print('Test loss',test_loss,'Test F1-Score',test_f1,'Confusion matrix:',test_cf)