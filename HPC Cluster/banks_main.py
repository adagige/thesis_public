import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'
import numpy as np
import yaml

import subprocess
import pandas as pd
import shutil

import argparse


# class MyDumper(yaml.Dumper):
#     def increase_indent(self, flow=False, indentless=False):
#         return super(MyDumper, self).increase_indent(flow, False)

# # Preserve the original structure by not using flow style for dictionaries
# def represent_dict_order(dumper, data):
#     return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

# yaml.add_representer(dict, represent_dict_order)



#banks = [20] #banks with more than X transactions 


def train_bank(bank, dataset):
   # dataset = 'HI-Large'
    dataset = dataset
    print(f'\n \n Training bank id {bank}', flush=True)
    #Remove the formatted transactions and the earlier saved data, ports etc. files
    # os.remove(f"Data/bank/{dataset}/AML/Small-HI/processed/data.pt")
    # os.remove(f"Data/bank/{dataset}/AML/Small-HI/processed/ports.pt")
    # os.remove(f"Data/bank/{dataset}/AML/Small-HI/processed/pre_filter.pt")
    # os.remove(f"Data/bank/{dataset}/AML/Small-HI/processed/pre_transform.pt")
    # os.remove(f"Data/bank/{dataset}/AML/formatted_transactions.csv")
    #create bank dataset in Data/bank/dataset/bank_id
    dataset_bank_path = f'Data/bank/bank{bank}/{dataset}/dataset{str(bank)}'
    dataset_path = f'Data/bank/bank{bank}/{dataset}/AML/{dataset}_Trans'
    if not os.path.exists(f'Data/bank/bank{bank}/{dataset}/AML'):
        os.makedirs(f'Data/bank/bank{bank}/{dataset}/AML')
    data = pd.read_csv(f'Data/AML/{dataset}_Trans.csv')
    data = data[(data['From Bank']==bank) | (data['To Bank']==bank)]
    data.to_csv(f'{dataset_bank_path}.csv',index=False)
    data.to_csv(f'{dataset_path}.csv',index=False)

    shutil.copyfile(f'configs/AML_{dataset}_Bank_template.yaml', f'configs/AML_{dataset}_Bank_{str(bank)}.yaml')
    
    out_dir = f"bank_results/{dataset}bank_{str(bank)}"
   # dataset.dir Data/bank/bank0/HI-Large
    common_params = f"out_dir {out_dir} dataset.dir Data/bank/bank{bank}/{dataset}"
    cfg_file = f'configs/AML_{dataset}_Bank_{str(bank)}.yaml'
    script = f"python main.py --cfg {cfg_file} --repeat 1 {common_params}"
    print(script, flush=True)
    subprocess.run(script, shell=True)

    #Evaluate the bank to get predictions out: 
    # script = f"python evaluate_bank.py --cfg configs/AML_HI-Large_Bank_0.yaml out_dir {out_dir} dataset.dir Data/bank/bank{bank}/{dataset}"
    # print(script, flush=True)
    # subprocess.run(script, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train bank model")
    parser.add_argument("--bank", type=int, required=True, help="Bank ID to process")
    parser.add_argument("--dataset", type=str, required=True, help="Bank ID to process")
    args = parser.parse_args()

    # Call the function with the parsed argument
    train_bank(args.bank, args.dataset)