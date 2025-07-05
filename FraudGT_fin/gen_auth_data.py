import pandas as pd
import numpy as np

banks = [0,11,12,20,27,70]
columns = [
    'Timestamp_x', 'From Bank', 'Account', 'To Bank', 'Account.1',
    'Amount Received_x', 'Receiving Currency', 'Amount Paid',
    'Payment Currency', 'Payment Format_x', 'Is Laundering_x', 'EdgeID',
    'from_id', 'to_id', 'Timestamp_y', 'Amount Sent', 'Sent Currency',
    'Amount Received_y', 'Received Currency', 'Payment Format_y',
    'Is Laundering_y', 'Edge_ID', 'Prediction', 'True_Label', 'Time_stamp',
    'Timestamp_new', 'predicted_label'
]


threshold = 0.5
df = pd.DataFrame(columns=columns)
folder ='bank_results_GatedGCN'# 'bank_results'#
gcn ='_GCN' #'' #  
dataset = 'HI-Large'
dataset_name = 'HI'

for gcn in ['_GCN','']:
    if gcn =='':
        folder = 'bank_results'
    else: 
        folder = 'bank_results_GatedGCN'

    for dataset_name in ['HI', 'LI']:
        if dataset_name == 'HI':
            dataset = 'HI-Large'
        else: 
            dataset = 'LI-Large'
        
        df = pd.DataFrame(columns=columns)
        for bank in banks:
            data= pd.read_csv(f'Data/bank/bank{bank}/{dataset}/dataset{bank}.csv')

            predictions_test = pd.read_csv(f'{folder}/{dataset}bank_{bank}/AML_{dataset}_Bank_{bank}{gcn}/predictions_test.csv').drop_duplicates('Edge_ID')
            predictions_val = pd.read_csv(f'{folder}/{dataset}bank_{bank}/AML_{dataset}_Bank_{bank}{gcn}/predictions_val.csv').drop_duplicates('Edge_ID')
            predictions_train = pd.read_csv(f'{folder}/{dataset}bank_{bank}/AML_{dataset}_Bank_{bank}{gcn}/predictions_train.csv').drop_duplicates('Edge_ID')

            #Save only the unique edge ids, to reduce the general computational complexity
            # predictions_test.to_csv(f'{folder}/{dataset}bank_{bank}/AML_{dataset}_Bank_{bank}{gcn}/predictions_test.csv', index=False)
            # predictions_val.to_csv(f'{folder}/{dataset}bank_{bank}/AML_{dataset}_Bank_{bank}{gcn}/predictions_val.csv',index=False)
            # predictions_train.to_csv(f'{folder}/{dataset}bank_{bank}/AML_{dataset}_Bank_{bank}{gcn}/predictions_train.csv',index=False)

            # print('predictions train shape is',predictions_train.shape, flush=True)
            # print('predictions train shape unique is',predictions_train.drop_duplicates('Edge_ID').shape, flush=True)
            # print('predictions val shape is',predictions_val.shape, flush=True)
            # print('predictions val shape unique is',predictions_val.drop_duplicates('Edge_ID').shape, flush=True)
            # print('predictions test shape is',predictions_test.shape, flush=True)
            # print('predictions test shape unique is',predictions_test.drop_duplicates('Edge_ID').shape, flush=True)

            predictions = pd.concat([predictions_test,predictions_val,predictions_train])

            predictions = pd.read_csv(f'{folder}/{dataset}bank_{bank}/AML_{dataset}_Bank_{bank}{gcn}/predictions_only_test.csv').drop_duplicates('Edge_ID')

            predictions = predictions.drop_duplicates(subset = 'Edge_ID')

            print('Bank ', bank, 'data size', data.shape, 'bank predictions', predictions.shape, flush=True)


            formatted = pd.read_csv(f'Data/bank/bank{bank}/{dataset}/AML/formatted_transactions.csv')
            df_merged =  data.merge(formatted, left_index=True, right_on='EdgeID')
            df_merged = df_merged.merge(predictions, left_on='EdgeID', right_on='Edge_ID')

            df_merged['predicted_label'] = df_merged['Prediction']  #1 / (1 + np.exp(-df_merged['Prediction']))

            df = pd.concat([df, df_merged], ignore_index=True)


        df.to_csv(f'{folder}/{dataset_name}_alldata_authorities_testdata.csv', index=False)
        df_to_send = df[df['predicted_label']>threshold]

        df_to_send.to_csv(f'{folder}/{dataset_name}_data_to_authorities_testdata.csv', index=False)

        column_mapping = {
            'Timestamp_x': 'Timestamp',
            'Amount Received_x': 'Amount Received',
            'Payment Format_x': 'Payment Format',
            'Is Laundering_x': 'Is Laundering'
            # Other columns remain unchanged
        }
        df_ori_data = df_to_send[['Timestamp_x', 'From Bank', 'Account', 'To Bank', 'Account.1',
            'Amount Received_x', 'Receiving Currency', 'Amount Paid',
            'Payment Currency', 'Payment Format_x', 'Is Laundering_x']]

        df_ori_data = df_ori_data.rename(columns=column_mapping)

        df_ori_data.to_csv(f'{folder}/{dataset_name}_ori_data_to_authorities_testdata.csv', index=False)

        print('Labeled as fraud size:',df_ori_data.shape, flush= True)