import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import torch
import os

def confusion_matrix_plot(data, title,save_to):
  threshold = 0.5
 # y_pred_prob = list(data['predicted_label'])
 
#  data['predicted_binary'] =  [1 if prob >= threshold else 0 for prob in y_pred_prob]

  data['predicted_binary'] = df_merged['predicted_label']
#  return data

  true_labels = data['True_Label'].to_numpy().astype(int)
  predicted_labels= data['predicted_binary'].to_numpy().astype(int)



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
  plt.title(f'{title} laundering')
  plt.savefig(f'sota_results/{save_to}.png')
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
  df.to_csv(f'sota_results/{save_to}.csv')
 # print(df)
#  print(classification_report(true_labels, predicted_labels, target_names=class_labels))


def patterns_transactions(data, patterns, save_to ):
    
    # Select only the relevant columns from patterns_HI
  #  patterns_reduced = patterns#[['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Laundering Type','Is Laundering']]

    # Merge the DataFrames
    merged_df = data.merge(
        patterns,  #
        how='left',  # Use 'left' to keep all rows from the first DataFrame
        left_on=['Timestamp_x','From Bank','To Bank', 'Account','Account.1','Amount Paid','Amount Received_x'],#[ 'From Bank', 'Account', 'To Bank', 'Account.1','Is Laundering_x'],  # Keys for the first DataFrame
        right_on=['Timestamp','From Bank','To Bank','Account','Account.1','Amount Paid','Amount Received'],#[ 'From Bank', 'Account', 'To Bank', 'Account.1','Is Laundering']  # Keys for the second DataFrame
    )

  #  threshold = 0.5
    patterns = merged_df.drop(columns=['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Is Laundering'])

    y_pred_prob = list(patterns['predicted_label'])
 
  #  patterns['predicted_binary'] =  [1 if prob >= threshold else 0 for prob in y_pred_prob]

    patterns['predicted_binary'] = patterns['predicted_label']

    patterns = patterns[patterns['Laundering Type'].notna()]
    patterns = patterns[['True_Label','predicted_binary','Laundering Type']]

    patterns['Correct'] = [1 if row['True_Label']==row['predicted_binary'] else 0 for i,row in patterns.iterrows()]
    patterns['False'] = [1 if row['True_Label']!=row['predicted_binary'] else 0 for i,row in patterns.iterrows()]

    patterns = patterns.groupby('Laundering Type')[['Correct', 'False']].sum()
    patterns['Correct_percent'] = patterns['Correct']/(patterns['Correct']+patterns['False'])
    patterns['False_percent'] = 1-patterns['Correct_percent']
   # f, ax = plt.subplots(figsize=(6, 6))
   # sns.heatmap(patterns,annot=True, ax=ax)
    
  
    patterns.to_csv(f'sota_results/{save_to}_patterns.csv')
    return 



patterns_large_HI = pd.read_csv('HI-Large_patterns.csv')
patterns_large_LI  = pd.read_csv('LI-Large_patterns.csv')
patterns_medium_HI = pd.read_csv('HI-Medium_patterns.csv')
patterns_medium_LI = pd.read_csv('LI-Medium_patterns.csv')
patterns_small_HI = pd.read_csv('HI-Small_patterns.csv')
patterns_small_LI = pd.read_csv('LI-Small_patterns.csv')



#all_banks_HI = pd.read_csv('../Data/all_bank_FraudGT/HI-Large/AML/HI_all_transactions.csv')


# HI_FraudGT = pd.read_csv('../bank_results/HI-Large_all_bank/AML_HI-Large_all_bank/HI_allpredictions_old.csv')
# LI_FraudGT = pd.read_csv('../bank_results/LI-Large_all_bank/AML_LI-Large_all_bank/LI_allpredictions.csv')


# confusion_matrix_plot(HI_FraudGT,'all banks HI FraudGT', 'all_banks_HI_FraudGT')
# confusion_matrix_plot(LI_FraudGT,'all banks LI FraudGT', 'all_banks_LI_FraudGT')

# results_folder = f'../bank_results/HI-Large_all_bank/AML_HI-Large_all_bank'

all_banks = {'HI_Large':{'results_folder':'../.results/AML-Large-HI-SOTA-return-Predictions/eval_test/AML-Large-HI-SOTA-return-Predictions',
                           'dataset':'../Data/AML/HI-Large/AML/HI-Large_Trans.csv',
                           'formatted':'../Data/AML/HI-Large/AML/',
                           'title':'Dataset Large-HI SOTA',
                           'save_to': 'LA_HI_SOTA',
                           'patterns':patterns_large_HI},
              'LI_Large':{'results_folder':'../.results/AML-Large-LI-SOTA-return-Predictions/eval_test/AML-Large-LI-SOTA-return-Predictions',
                           'dataset':'../Data/AML/LI-Large/AML/LI-Large_Trans.csv',
                           'formatted':'../Data/AML/LI-Large/AML/',
                           'title':'Dataset Large-LI SOTA',
                           'save_to': 'LA_LI_SOTA',
                           'patterns':patterns_large_LI},
              'HI_Medium':{'results_folder':'../.results/AML-Medium-HI-SOTA-return-Predictions/eval_test/AML-Medium-HI-SOTA-return-Predictions',
                           'dataset':'../Data/AML/HI-Medium/AML/HI-Medium_Trans.csv',
                           'formatted':'../Data/AML/HI-Medium/AML/',
                           'title':'Dataset Medium-HI SOTA',
                           'save_to': 'ME_HI_SOTA',
                           'patterns':patterns_medium_HI},
              'LI_Medium':{'results_folder':'../.results/AML-Medium-LI-SOTA-return-Predictions/eval_test/AML-Medium-LI-SOTA-return-Predictions',
                           'dataset':'../Data/AML/LI-Medium/AML/LI-Medium_Trans.csv',
                           'formatted':'../Data/AML/LI-Medium/AML/',
                           'title':'Dataset Medium-LI SOTA',
                           'save_to': 'ME_LI_SOTA',
                           'patterns':patterns_medium_LI},
              'HI_Small':{'results_folder':'../.results/AML-Small-HI-SOTA-return-Predictions/eval_test/AML-Small-HI-SOTA-return-Predictions',
                           'dataset':'../Data/AML/HI-Small/AML/HI-Small_Trans.csv',
                           'formatted':'../Data/AML/HI-Small/AML/',
                           'title':'Dataset Small-HI SOTA',
                           'save_to': 'SM_HI_SOTA',
                           'patterns':patterns_small_HI},
              'LI_Small':{'results_folder':'../.results/AML-Small-LI-SOTA-return-Predictions/eval_test/AML-Small-LI-SOTA-return-Predictions',
                           'dataset':'../Data/AML/LI-Small/AML/LI-Small_Trans.csv',
                           'formatted':'../Data/AML/LI-Small/AML/',
                           'title':'Dataset Small-LI SOTA',
                           'save_to': 'SM_LI_SOTA',
                           'patterns':patterns_small_LI},
            }



for model in all_banks.values():
    print('Running model', model, flush=True)
    results_folder = model['results_folder']
    dataset = model['dataset']
    formatted_folder = model['formatted']
    patterns = model['patterns']

    if os.path.exists(f'{results_folder}/predictions_only_test_all.csv'):
      df_merged = pd.read_csv(f'{results_folder}/predictions_only_test_all.csv')

    else:# make the predictions file: 

      data= pd.read_csv(dataset)
      predictions_test = pd.read_csv(f'{results_folder}/predictions_only_test.csv').drop_duplicates('Edge_ID')
    # predictions_val = pd.read_csv(f'{results_folder}/predictions_val.csv').drop_duplicates('Edge_ID')
      #predictions_train = pd.read_csv(f'{results_folder}/predictions_train.csv').drop_duplicates('Edge_ID')

      predictions = predictions_test
    #  predictions = pd.concat([predictions_test,predictions_val,predictions_train])

      predictions = predictions.drop_duplicates('Edge_ID')
      
      print('Predictions size:', predictions.shape, flush=True )
      print('Full data size:', data.shape, flush=True )

      formatted = pd.read_csv(f'{formatted_folder}/formatted_transactions.csv')

      df_merged =  data.merge(formatted, left_index=True, right_on='EdgeID')
      df_merged = df_merged.merge(predictions, left_on='EdgeID', right_on='Edge_ID')

   #   df_merged['predicted_label'] = 1 / (1 + np.exp(-df_merged['Prediction']))
      df_merged['predicted_label'] = df_merged['Prediction']

      df_merged.to_csv(f'{results_folder}/predictions_only_test_all.csv', index=False)

      

    confusion_matrix_plot(df_merged, title=model['title'], save_to=f'{model['save_to']}')

    patterns_transactions(df_merged, patterns, save_to=f'{model['save_to']}')



# all_banks_HI = pd.read_csv('../Data/all_bank/HI-Large/AML/HI_all_transactions.csv')

#print(all_banks_HI.shape, flush=True)

# all_banks_HI = pd.read_csv('../Data/all_bank_FraudGT/HI-Large/AML/HI-Large_Trans.csv')

# print(all_banks_HI.shape, flush=True)

# results_folder = f'../bank_results/HI-Large_all_bank/AML_HI-Large_all_bank'
# #data= pd.read_csv(f'{data_folder}/{dataset}_Trans.csv')
# predictions_test = pd.read_csv(f'{results_folder}/predictions_test.csv')
# predictions_val = pd.read_csv(f'{results_folder}/predictions_val.csv')
# predictions_train = pd.read_csv(f'{results_folder}/predictions_train.csv')

# print('Train before removing duplicates',predictions_train.shape, flush=True)

# predictions_train = predictions_train.drop_duplicates(subset = 'Edge_ID')
# predictions_val = predictions_val.drop_duplicates(subset = 'Edge_ID')
# predictions_test = predictions_test.drop_duplicates(subset = 'Edge_ID')


# print('test',predictions_test.shape, flush=True)
# print('Val',predictions_val.shape, flush=True)
# print('Train',predictions_train.shape, flush=True)

# all_unique_edgeID =  set(predictions_test['Edge_ID']).union(set(predictions_val['Edge_ID']), set(predictions_train['Edge_ID']))

# print('Number of unique edgeids:', len(all_unique_edgeID))

# data_processed = torch.load('../Data/all_bank_FraudGT/HI-Large/AML/Large-HI/processed/data.pt')


# print(data_processed, flush=True)



# data_BFGT_FraudGT = pd.read_csv('../Data/authBFGT_FraudGT/HI-Large/AML/HI-Large_Trans.csv')
# pred_BFGT_FraudGT = pd.read_csv('HI_BFGT_FraudGT_allpredictions_all.csv')

# print('pred_BFGT_FraudGT',pred_BFGT_FraudGT.shape,flush=True)

# print('data_HI Large BFGT Fraud', data_BFGT_FraudGT.shape, flush=True)

# BFGT_test = pd.read_csv('../bank_results/HI-Large_auth_FraudGT/AML_HI-Large_auth_BFGT_FraudGT/predictions_test.csv')
# BFGT_val = pd.read_csv('../bank_results/HI-Large_auth_FraudGT/AML_HI-Large_auth_BFGT_FraudGT/predictions_val.csv')
# BFGT_train = pd.read_csv('../bank_results/HI-Large_auth_FraudGT/AML_HI-Large_auth_BFGT_FraudGT/predictions_train.csv')

# print('test',BFGT_test.shape, flush=True)
# print('Val',BFGT_val.shape, flush=True)
# print('Train',BFGT_train.shape, flush=True)