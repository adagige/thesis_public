import pandas as pd
import numpy as np

columns = [
    'Timestamp_x', 'From Bank', 'Account', 'To Bank', 'Account.1',
    'Amount Received_x', 'Receiving Currency', 'Amount Paid',
    'Payment Currency', 'Payment Format_x', 'Is Laundering_x', 'EdgeID',
    'from_id', 'to_id', 'Timestamp_y', 'Amount Sent', 'Sent Currency',
    'Amount Received_y', 'Received Currency', 'Payment Format_y',
    'Is Laundering_y', 'Edge_ID', 'Prediction', 'True_Label', 'Time_stamp',
    'Timestamp_new', 'predicted_label'
]


# threshold = 0.5
# df = pd.DataFrame(columns=columns)
# folder ='bank_results'# 'bank_results_GatedGCN/'#
# gcn = '' #'_GCN'
# dataset = 'HI-Large'
dataset_name = 'HI'
dataset = 'HI-Large'
GCN = '_GCN'# ''#  '_GCN'#

# The below are for the fraudgt banks 
data_folder = f'Data/all_bank_GCN/{dataset}/AML'
results_folder = f'bank_results_GatedGCN/{dataset}_all_bank/AML_{dataset}_all_bank{GCN}'
data= pd.read_csv(f'{data_folder}/{dataset}_Trans.csv')
predictions_test = pd.read_csv(f'{results_folder}/predictions_test.csv')
predictions_val = pd.read_csv(f'{results_folder}/predictions_val.csv')
predictions_train = pd.read_csv(f'{results_folder}/predictions_train.csv')
predictions = pd.concat([predictions_test,predictions_val,predictions_train])

formatted = pd.read_csv(f'{data_folder}/formatted_transactions.csv')
# The below are for the GCN banks 

# data= pd.read_csv(f'Data/auth_GCN/{dataset}/AML/{dataset}_Trans.csv')

# folder = 'bank_results_GatedGCN/HI-Large_auth_GCN/AML_HI-Large_auth_BGCN_GCN'
# formatted = pd.read_csv(f'Data/auth_GCN/{dataset}/AML/formatted_transactions.csv')
# data= pd.read_csv(f'Data/auth_GCN/{dataset}/AML/{dataset}_Trans.csv')
# predictions = pd.read_csv(f'{folder}/predictions.csv')


df_merged =  data.merge(formatted, left_index=True, right_on='EdgeID')
df_merged = df_merged.merge(predictions, left_on='EdgeID', right_on='Edge_ID')

df_merged['predicted_label'] = 1 / (1 + np.exp(-df_merged['Prediction']))

#FraudGT: 
# df_merged.to_csv(f'bank_results/HI-Large_auth/AML_HI-Large_auth/{dataset_name}_alldata_authorities.csv', index=False)
#GCN:
#df_merged.to_csv(f'bank_results_GatedGCN/HI-Large_auth/AML_HI-Large_auth_BGCNfraud/{dataset_name}_alldata_authorities.csv', index=False)
df_merged.to_csv(f'{results_folder}/{dataset_name}_allpredictions.csv', index=False)





