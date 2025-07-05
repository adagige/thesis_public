import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


datasets = {'Hi-Small':'bank_analysis/Hi-Small_banks_data.csv', 
            'Li-Small':'bank_analysis/Li-Small_banks_data.csv',
            'Hi-Medium':'bank_analysis/Hi-Medium_banks_data.csv', 
            'Li-Medium': 'bank_analysis/Li-Medium_banks_data.csv',
            'Hi-Large': 'bank_analysis/Hi-Large_banks_data.csv', 
            'Li-Large': 'bank_analysis/Li-Large_banks_data.csv', 
           }

for dataset in datasets.keys():
    print(f'Currently processing dataset {dataset}',flush=True)
    data = pd.read_csv(datasets[dataset])
    print(f'Number of banks in dataset {dataset} are {data.shape[0]}', flush=True)
    trans_more = data[data['transaction_count']>5]
    print(f'Number of banks with more than 5 transactions in dataset {dataset} are {trans_more.shape[0]}', flush=True)
    trans_more = data[data['transaction_count']>1000]
    print(f'Number of banks with more than 1000 transactions in dataset {dataset} are {trans_more.shape[0]}', flush=True)
    trans_more = data[data['transaction_count']>10000]
    print(f'Number of banks with more than 10,000 transactions in dataset {dataset} are {trans_more.shape[0]}', flush=True)
    trans_more = data[data['transaction_count']>100000]
    print(f'Number of banks with more than 100,000 transactions in dataset {dataset} are {trans_more.shape[0]}', flush=True)
    trans_more = data[data['transaction_count']>1000000]
    print(f'Number of banks with more than 1,000,000 transactions in dataset {dataset} are {trans_more.shape[0]}', flush=True)

    print(f'Smallest money laundering in {dataset} are {min(data['money_laundering_fraction'])*100}', flush=True)
    print(f'Highest money laundering in {dataset} are {max(data['money_laundering_fraction'])*100}', flush=True)
    print(f'Average money laundering in {dataset} are {round(np.mean(data['money_laundering_fraction'])*100,8)}', flush=True)
    print(f"The large banks (more than 1 mio.) have money laundery distribution: {trans_more['money_laundering_fraction']*100}", flush=True)
    print(f"size of largest banks:{trans_more['transaction_count']}", flush = True)
    plt.hist(data['money_laundering_fraction'])
    plt.savefig(f'bank_analysis/{dataset}_ML_fraction.png')

