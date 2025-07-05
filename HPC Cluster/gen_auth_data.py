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
dataset = 'HI-Large'
dataset_name = 'HI'
for bank in banks:
    data= pd.read_csv(f'Data/bank/bank{bank}/{dataset}/dataset{bank}.csv')
    predictions = pd.read_csv(f'bank_results/{dataset}bank_{bank}/AML_{dataset}_Bank_{bank}/predictions.csv')
    formatted = pd.read_csv(f'Data/bank/bank{bank}/{dataset}/AML/formatted_transactions.csv')
    df_merged =  data.merge(formatted, left_index=True, right_on='EdgeID')
    df_merged = df_merged.merge(predictions, left_index=True, right_on='Edge_ID')

    df_merged['predicted_label'] = 1 / (1 + np.exp(-df_merged['Prediction']))

    df = pd.concat([df, df_merged], ignore_index=True)


df.to_csv(f'bank_results/{dataset_name}_alldata_authorities.csv', index=False)
df_to_send = df[df['predicted_label']>threshold]

df_to_send.to_csv(f'bank_results/{dataset_name}_data_to_authorities.csv', index=False)



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

df_ori_data.to_csv(f'bank_results/{dataset_name}_ori_data_to_authorities.csv', index=False)

print('Labeled as fraud size:',df_ori_data.shape, flush= True)