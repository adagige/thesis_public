import pandas as pd
import matplotlib.pyplot as plt


def bank_metadata(data,DatasetName):
    all_banks = list(set(data['From Bank'].unique()).union(set(data['To Bank'].unique()))) #Get all unique banks
    in_degree = []
    out_degree = []
    transactions_amount = []
    in_degree_sum = []
    out_degree_sum = []
    transactions_sum = []
    money_laundering_frac = []

    for bank in all_banks: 
        bank_data = data[(data['From Bank']==bank) | (data['To Bank']==bank)]
        trans_to =  bank_data[bank_data['To Bank']==bank].shape[0] #Basically the in-degree
        trans_from = bank_data[bank_data['From Bank']==bank].shape[0] #Basically the out-degree
        trans_total = bank_data.shape[0] #Total amount of transactions through that bank 
        trans_to_sum = bank_data[bank_data['To Bank']==bank]['Amount Received'].sum() #Money received at the bank 
        trans_from_sum = bank_data[bank_data['From Bank']==bank]['Amount Paid'].sum() # Money send from the bank
        money_laundering = bank_data['Is Laundering'].sum()/ trans_total
        trans_sum = trans_to_sum + trans_from_sum

        #Append calculated data to the lists generated:
        in_degree.append(trans_to)
        out_degree.append(trans_from)
        transactions_amount.append(trans_total)
        in_degree_sum.append(trans_to_sum)
        out_degree_sum.append(trans_from_sum)
        transactions_sum.append(trans_sum)
        money_laundering_frac.append(money_laundering)
        
    all_data = pd.DataFrame({
        'Bank_id' : all_banks,
        'in_degree' : in_degree,
        'out_degree' : out_degree,
        'transaction_count' : transactions_amount,
        'in_degree_sum' : in_degree_sum,
        'out_degree_sum': out_degree_sum,
        'transaction_sum': transactions_sum,
        'money_laundering_fraction': money_laundering_frac        
    })

    #Plotting - All the plots 
    # List of columns to plot
    columns_to_plot = ['in_degree', 'out_degree', 'transaction_count', 'in_degree_sum', 'out_degree_sum', 'transaction_sum']

    # Plotting the distributions
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    fig.suptitle(f'Log-Scaled Distribution of banks in dataset {DatasetName}')

 
    for ax, column in zip(axes.flatten(), columns_to_plot):
        n, bins, patches = ax.hist(all_data[column], bins=20, edgecolor='k', log=True)
        ax.set_title(f'Log-Scaled Distribution of {column} ')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'bank_analysis/{DatasetName}_bank_figure.png')


    return all_data




datasets = {'Hi-Small':'../Data/AML/HI-Small_Trans.csv', 
           'Li-Small':'../Data/AML/LI-Small_Trans.csv',
           'Hi-Medium':'../Data/AML/HI-Medium_Trans.csv', 
           'Li-Medium': '../Data/AML/LI-Medium_Trans.csv',
           'Hi-Large': '../Data/AML/HI-Large_Trans.csv', 
            'Li-Large': '../Data/AML/LI-Large_Trans.csv', 
           }



for dataset in datasets.keys():
    print(f'Currently processing dataset {dataset}',flush=True)
    data = pd.read_csv(datasets[dataset])
    all_data = bank_metadata(data, dataset)
    print(f'Saving bank dataset for {dataset}',flush=True)
    all_data.to_csv(f'bank_analysis/{dataset}_banks_data.csv', index=False)
    
print('All datasets analysed and data can be found in bank_analysis',flush=True)