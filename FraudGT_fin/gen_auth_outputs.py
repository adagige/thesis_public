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
dataset_name = 'HI'
dataset = 'HI-Large'
auth_model = 'FraudGT' #GCN FraudGT
bank_model = 'BGCN' #BGCN BFGT
all_trans = True

for dataset in ['HI-Large','LI-Large']:
    if dataset == 'HI-Large':
        dataset_name = 'HI'
    else: 
        dataset_name = 'LI'
    for auth_model in ['FraudGT', 'GCN']:
        for bank_model in ['BGCN', 'BFGT']:
            if bank_model == 'BFGT':
                folder ='bank_results'# 'bank_results_GatedGCN/'#
            elif bank_model == 'BGCN':
                folder = 'bank_results_GatedGCN'

            data = pd.read_csv(f'Data/auth{bank_model}_{auth_model}/{dataset}/AML/{dataset}_Trans.csv')
            formatted = pd.read_csv(f'Data/auth{bank_model}_{auth_model}/{dataset}/AML/formatted_transactions.csv')

            if all_trans:
                predictions_test = pd.read_csv(f'{folder}/{dataset}_auth_{auth_model}/AML_{dataset}_auth_{bank_model}_{auth_model}/predictions_test.csv',on_bad_lines='skip').drop_duplicates('Edge_ID')
                predictions_val = pd.read_csv(f'{folder}/{dataset}_auth_{auth_model}/AML_{dataset}_auth_{bank_model}_{auth_model}/predictions_val.csv',on_bad_lines='skip').drop_duplicates('Edge_ID')
                predictions_train = pd.read_csv(f'{folder}/{dataset}_auth_{auth_model}/AML_{dataset}_auth_{bank_model}_{auth_model}/predictions_train.csv',on_bad_lines='skip').drop_duplicates('Edge_ID')
                predictions = pd.concat([predictions_test,predictions_val,predictions_train])
                predictions = predictions.drop_duplicates('Edge_ID')

                #Save the filtered dataset to not gain to large datasets when the evaluation i run several times: 
                predictions_test.to_csv(f'{folder}/{dataset}_auth_{auth_model}/AML_{dataset}_auth_{bank_model}_{auth_model}/predictions_test.csv',index=False)
                predictions_val.to_csv(f'{folder}/{dataset}_auth_{auth_model}/AML_{dataset}_auth_{bank_model}_{auth_model}/predictions_val.csv',index = False)
                predictions_train.to_csv(f'{folder}/{dataset}_auth_{auth_model}/AML_{dataset}_auth_{bank_model}_{auth_model}/predictions_train.csv',index = False)
                

            else:
                predictions = pd.read_csv(f'{folder}/{dataset}_auth_{auth_model}/AML_{dataset}_auth_{bank_model}_{auth_model}/predictions_only_test.csv')
                predictions = predictions.drop_duplicates('Edge_ID')
                # predictions['Edge_ID'] = predictions['Edge_ID'].astype(int)

            # data['EdgeID'] = data['EdgeID'].astype(int)

            print('Dataset and models ', f'{dataset} {bank_model} {auth_model}', 'data size', data.shape, formatted.shape ,'predictions shape', predictions.shape, 'difference:', predictions.shape[0]-data.shape[0],flush=True)

            df_merged =  data.merge(formatted, left_index=True, right_on='EdgeID')
            df_merged = df_merged.merge(predictions, how='left',left_on='EdgeID', right_on='Edge_ID')

            print(df_merged['Prediction'].isna().sum(), flush = True)

            df_merged['Prediction'] = df_merged['Prediction'].fillna(10)

            df_merged['predicted_label'] = df_merged['Prediction']# 1 / (1 + np.exp(-df_merged['Prediction']))

            #FraudGT: 
            # df_merged.to_csv(f'bank_results/HI-Large_auth/AML_HI-Large_auth/{dataset_name}_alldata_authorities.csv', index=False)
            #GCN:
            #df_merged.to_csv(f'bank_results_GatedGCN/HI-Large_auth/AML_HI-Large_auth_BGCNfraud/{dataset_name}_alldata_authorities.csv', index=False)

            if all_trans:
                df_merged.to_csv(f'{folder}/{dataset}_auth_{auth_model}/AML_{dataset}_auth_{bank_model}_{auth_model}/{dataset_name}_{bank_model}_{auth_model}_allpredictions_all.csv', index=False)
            else:
                df_merged.to_csv(f'{folder}/{dataset}_auth_{auth_model}/AML_{dataset}_auth_{bank_model}_{auth_model}/{dataset_name}_{bank_model}_{auth_model}_allpredictions_test.csv', index=False)





