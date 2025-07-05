import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

def confusion_matrix_plot(data, title,save_to):
  threshold = 0.5
  y_pred_prob = list(data['predicted_label'])
 
  data['predicted_binary'] =  data['Prediction']#[1 if prob >= threshold else 0 for prob in y_pred_prob]
#  return data

  true_labels = data['True_Label']
  predicted_labels= data['predicted_binary']


  conf_matrix = confusion_matrix(true_labels, predicted_labels)
  class_labels = ['Not-Fraud', 'Fraud']

  total = conf_matrix.sum()
  annot = np.empty_like(conf_matrix).astype(str)
  for i in range(conf_matrix.shape[0]):
      for j in range(conf_matrix.shape[1]):
          count = conf_matrix[i, j]
          percentage = count / total * 100
          annot[i, j] = f'{count:,}\n({percentage:.1f}%)'

  # Create a heatmap using seaborn
  plt.figure(figsize=(6, 4))
  sns.heatmap(conf_matrix, annot=annot, fmt='', cmap='PuRd', xticklabels=class_labels, yticklabels=class_labels)

  # Add labels, title, and display the plot
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.title(f'{title} laundering', pad=20)
  plt.savefig(f'auth_results_test/{save_to}.png')
  plt.show()
  # Calculate precision, recall, and F1-score
  precision = precision_score(true_labels, predicted_labels, average='binary')
  recall = recall_score(true_labels, predicted_labels, average='binary')
  f1 = f1_score(true_labels, predicted_labels, average='binary')

  print(f"Precision: {precision:.2f}")
  print(f"Recall: {recall:.2f}")
  print(f"F1-Score: {f1:.2f}")

  # Alternatively, print a full classification report
  print("\nClassification Report:")
  report = classification_report(true_labels, predicted_labels, target_names=class_labels, output_dict=True)
  df = pd.DataFrame(report).transpose()
  df.to_csv(f'auth_results_test/{save_to}.csv')
 # print(df)
#  print(classification_report(true_labels, predicted_labels, target_names=class_labels))


def patterns_transactions(data, patterns, save_to ):
    
    # Select only the relevant columns from patterns_HI
  #  patterns_reduced = patterns#[['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Laundering Type','Is Laundering']]

    # Merge the DataFrames
    merged_df = data.merge(
        patterns,  #
        how='left',  # Use 'left' to keep all rows from the first DataFrame
        left_on=['Timestamp','From Bank','To Bank', 'Account','Account.1','Amount Paid','Amount Received'],#[ 'From Bank', 'Account', 'To Bank', 'Account.1','Is Laundering_x'],  # Keys for the first DataFrame
        right_on=['Timestamp','From Bank','To Bank','Account','Account.1','Amount Paid','Amount Received'],#[ 'From Bank', 'Account', 'To Bank', 'Account.1','Is Laundering']  # Keys for the second DataFrame
    )

    threshold = 0.5
   # patterns = merged_df.drop(columns=['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Is Laundering'])
    patterns = merged_df
    # y_pred_prob = list(patterns['predicted_label'])
 
    patterns['predicted_binary'] = patterns['Prediction']# [1 if prob >= threshold else 0 for prob in y_pred_prob]

    patterns = patterns[patterns['Laundering Type'].notna()]
    patterns = patterns[['True_Label','predicted_binary','Laundering Type']]

    patterns['Correct'] = [1 if row['True_Label']==row['predicted_binary'] else 0 for i,row in patterns.iterrows()]
    patterns['False'] = [1 if row['True_Label']!=row['predicted_binary'] else 0 for i,row in patterns.iterrows()]

    patterns = patterns.groupby('Laundering Type')[['Correct', 'False']].sum()
    patterns['Correct_percent'] = patterns['Correct']/(patterns['Correct']+patterns['False'])
    patterns['False_percent'] = 1-patterns['Correct']
   # f, ax = plt.subplots(figsize=(6, 6))
   # sns.heatmap(patterns,annot=True, ax=ax)
    
  
    patterns.to_csv(f'auth_results_test/{save_to}_patterns.csv')
    return 


BF_AF_HI = {'data': pd.read_csv('../bank_results/HI-Large_auth_FraudGT/AML_HI-Large_auth_BFGT_FraudGT/HI_BFGT_FraudGT_allpredictions_test.csv'), 
            'name': 'HI_BFGT_FraudGT', 
            'title': 'High bank FraudGT Authority FraudGT'}
BF_AF_LI = {'data': pd.read_csv('../bank_results/LI-Large_auth_FraudGT/AML_LI-Large_auth_BFGT_FraudGT/LI_BFGT_FraudGT_allpredictions_test.csv'),'name':'LI_BFGT_FraudGT', 'title': 'Low bank FraudGT Authority FraudGT'}

BF_AG_HI = {'data': pd.read_csv('../bank_results/HI-Large_auth_GCN/AML_HI-Large_auth_BFGT_GCN/HI_BFGT_GCN_allpredictions_test.csv'), 'name': 'HI_BFGT_GCN', 'title': 'High bank FraudGT Authority GatedGCN'}
BF_AG_LI = {'data': pd.read_csv('../bank_results/LI-Large_auth_GCN/AML_LI-Large_auth_BFGT_GCN/LI_BFGT_GCN_allpredictions_test.csv'), 'name': 'LI_BFGT_GCN','title': 'Low bank FraudGT Authority GatedGCN'}

BG_AG_HI = {'data': pd.read_csv('../bank_results_GatedGCN/HI-Large_auth_GCN/AML_HI-Large_auth_BGCN_GCN/HI_BGCN_GCN_allpredictions_test.csv'), 'name': 'HI_BGCN_GCN','title': 'High bank GatedGCN Authority GatedGCN'}
BG_AG_LI = {'data': pd.read_csv('../bank_results_GatedGCN/LI-Large_auth_GCN/AML_LI-Large_auth_BGCN_GCN/LI_BGCN_GCN_allpredictions_test.csv'), 'name': 'LI_BGCN_GCN','title': 'Low bank GatedGCN Authority GatedGCN'}

BG_AF_HI = {'data': pd.read_csv('../bank_results_GatedGCN/HI-Large_auth_FraudGT/AML_HI-Large_auth_BGCN_FraudGT/HI_BGCN_FraudGT_allpredictions_test.csv'), 'name': 'HI_BGCN_FraudGT','title': 'High bank GatedGCN Authority FraudGT'}
BG_AF_LI = {'data': pd.read_csv('../bank_results_GatedGCN/LI-Large_auth_FraudGT/AML_LI-Large_auth_BGCN_FraudGT/LI_BGCN_FraudGT_allpredictions_test.csv'), 'name': 'LI_BGCN_FraudGT','title': 'Low bank GatedGCN Authority FraudGT'}

patterns_HI = pd.read_csv('HI-Large_patterns.csv')
patterns_LI  = pd.read_csv('LI-Large_patterns.csv')

all_banks_HI = pd.read_csv('../Data/all_bank/HI-Large/AML/HI_all_transactions.csv') #['Timestamp_x', 'From Bank', 'Account', 'To Bank', 'Account.1',
      #  'Amount Received_x', 'Receiving Currency', 'Amount Paid',
      #  'Payment Currency', 'Payment Format_x', 'Is Laundering_x', 'EdgeID',
      #  'from_id', 'to_id', 'Timestamp_y', 'Amount Sent', 'Sent Currency',
      #  'Amount Received_y', 'Received Currency', 'Payment Format_y',
      #  'Is Laundering_y']

all_banks_LI = pd.read_csv('../Data/all_bank/LI-Large/AML/LI_all_transactions.csv')


all_banks_HI = all_banks_HI[['Timestamp_x', 'From Bank', 'Account', 'To Bank', 'Account.1',
       'Amount Received_x', 'Receiving Currency', 'Amount Paid',
       'Payment Currency', 'Payment Format_x', 'Is Laundering_x', 'EdgeID',
       'from_id', 'to_id','Amount Sent', 'Sent Currency','Received Currency']].rename(columns={'Timestamp_x':'Timestamp','Amount Received_x':'Amount Received','Payment Format_x':'Payment Format','Is Laundering_x':'Is Laundering'})

models_HI_list = [BF_AF_HI, BF_AG_HI,BG_AF_HI, BG_AG_HI]
models_LI_list = [BF_AF_LI, BF_AG_LI,BG_AF_LI, BG_AG_LI]

# models_HI_list = [BG_AG_HI,BF_AF_HI,BF_AG_HI]
# models_LI_list = [BG_AG_LI,BF_AF_LI,BF_AG_LI]


for model in models_HI_list: 
    model_data = model['data']
    model_name = model['name']
   # print('Columns are named:',all_banks_HI.columns, flush=True)

    model_data = model_data[['Timestamp_x', 'From Bank', 'Account', 'To Bank', 'Account.1',
       'Amount Received_x', 'Receiving Currency', 'Amount Paid',
       'Payment Currency', 'Payment Format_x', 'Is Laundering_x', 'EdgeID',
       'from_id', 'to_id','Amount Sent', 'Sent Currency','Received Currency', 'Edge_ID', 'Prediction', 'True_Label',
       'predicted_label']].rename(columns={'Timestamp_x':'Timestamp','Amount Received_x':'Amount Received','Payment Format_x':'Payment Format','Is Laundering_x':'Is Laundering'})


    model_data = model_data.drop_duplicates('Edge_ID')

    confusion_matrix_plot(model_data,save_to=model_name, title=model['title'])

    patterns_transactions(model_data, patterns_HI,model_name )



    #concat data 
    # merged = all_banks_HI.merge(model_data,how='left', left_on='EdgeID', right_on='EdgeID')

    # merged = merged.drop_duplicates('EdgeID')

    # merged = merged[(merged['From Bank_x']!= 70 )& (merged['To Bank_x']!= 70)]

    # print('merged new size:', merged.shape, flush=True)
  

    # threshold = 0.5

    # merged['not received'] = merged['predicted_label'].isna().astype(int)
    # merged['predicted fraud'] = (merged['predicted_label'] >= threshold).astype(int)
    # merged['predicted not-fraud'] = (merged['predicted_label'] < threshold).astype(int)


    # merged['True Label'] = np.where(
    #     merged['not received'] == 1, 
    #     merged['Is Laundering_x'], 
    #     merged['True_Label']
    # )

    # table = merged.groupby(['True Label'])[['not received','predicted fraud', 'predicted not-fraud']].sum().T.rename(columns={0:'Not-Fraud', 1:'Fraud'})#.to_csv('sample_auth.csv')

    # table.to_csv(f'auth_results_test/{model_name}.csv')

    # plt.figure(figsize=(6, 4))
    # sns.heatmap(table, annot=True, fmt=",.0f", cmap='PuRd')#, xticklabels=class_labels, yticklabels=class_labels)
    # plt.yticks(rotation=0)
    # plt.title(f'{model['title']}')
    # plt.savefig(f'auth_results_test/{model_name}.png',bbox_inches = "tight")
    # plt.close()




for model in models_LI_list: 
    model_data = model['data']
    model_name = model['name']

    model_data = model_data[['Timestamp_x', 'From Bank', 'Account', 'To Bank', 'Account.1',
       'Amount Received_x', 'Receiving Currency', 'Amount Paid',
       'Payment Currency', 'Payment Format_x', 'Is Laundering_x', 'EdgeID',
       'from_id', 'to_id','Amount Sent', 'Sent Currency','Received Currency', 'Edge_ID', 'Prediction', 'True_Label',
       'predicted_label']].rename(columns={'Timestamp_x':'Timestamp','Amount Received_x':'Amount Received','Payment Format_x':'Payment Format','Is Laundering_x':'Is Laundering'})

    model_data = model_data.drop_duplicates('Edge_ID')

    confusion_matrix_plot(model_data,save_to=model_name, title=model['title'])

    patterns_transactions(model_data, patterns_LI,model_name )

    # #concat data 
    # merged = all_banks_LI.merge(model_data,how='left', left_on='EdgeID', right_on='EdgeID')

    # merged = merged.drop_duplicates('EdgeID')

    # # merged = merged[(merged['From Bank_x']!= 70 )& (merged['To Bank_x']!= 70)]

    # # print('merged new size:', merged.shape, flush=True)

    # # merged['not received'] = [1 if pd.isna(row['predicted_label']) else 0 for i,row in merged.iterrows()]
    # # merged['True_Label'] = [row['Is Laundering_x'] if row['not received'] == 1 else row['True_Label'] for i,row in merged.iterrows()]
    # # merged['predicted fraud'] =  [1 if row['predicted_label'] >= 0.5 else 0 for i,row in merged.iterrows()]
    # # merged['predicted not-fraud'] =  [1 if row['predicted_label'] < 0.5 else 0 for i,row in merged.iterrows()]

    # threshold = 0.5

    # merged['not received'] = merged['predicted_label'].isna().astype(int)
    # merged['predicted fraud'] = (merged['predicted_label'] >= threshold).astype(int)
    # merged['predicted not-fraud'] = (merged['predicted_label'] < threshold).astype(int)


    # merged['True Label'] = np.where(
    #     merged['not received'] == 1, 
    #     merged['Is Laundering_x'], 
    #     merged['True_Label']
    # )

    # table = merged.groupby(['True Label'])[['not received','predicted fraud', 'predicted not-fraud']].sum().T.rename(columns={0:'Not-Fraud', 1:'Fraud'})#.to_csv('sample_auth.csv')

    # table.to_csv(f'auth_results_test/{model_name}.csv')

    # plt.figure(figsize=(6, 4))
    # sns.heatmap(table, annot=True,  fmt=",.0f", cmap='PuRd')#, xticklabels=class_labels, yticklabels=class_labels)
    # plt.yticks(rotation=0)
    # plt.title(f'{model['title']}')
    # plt.savefig(f'auth_results_test/{model_name}.png',bbox_inches = "tight")

    # plt.close()
