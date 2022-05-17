# Model Evaluation Exercises
from statistics import harmonic_mean
import pandas as pd
import numpy as np
import statistics as stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score

# The confusion matrix:
# |               | pred dog   | pred cat   |
# |:------------  |-----------:|-----------:|
# | actual dog    |         46 |         7  |
# | actual cat    |         13 |         34 |

#2
# I'm assuming this model is designed to identify cats
# FP: The model predicted a cat, but it was actually a dog
# FN: The model predicted it wasn't a cat, but it was
# This model has higher precision (doesn't often say it is a cat when it isn't)
#    than Sensitivity (it often says it's not a cat when it is)

#3
rd_df = pd.read_csv('https://ds.codeup.com/data/c3.csv')
#shape: 200x4
#columns: actual, model1, model2, model3
#3a) Want to identify as many ducks that have the defects as possible:
#  ANS: For this I want to minimize false negatives (missing a defected duck), 
#           so I want a model with high recall/sensitivity
pd.crosstab(rd_df.model1,rd_df.actual)
# actual     Defect  No Defect
# model1                      
# Defect          8          2
# No Defect       8        182

## IMPROVEMENTS TO MAKE:
# drop into a function
# put results in df
# print df

for i in rd_df.columns[1:]:
    #outputs in different order.  
    # I want postivie to equal "no defect", so put that in first label
    tn, fp, fn, tp = confusion_matrix(rd_df.actual,rd_df[i],labels=('No Defect','Defect')).ravel()
    print(f'\n\033[1m{i}\033[0m')
    print(f'TP: {tp}   FP: {fp}')
    print(f'FN: {fn}   TN: {tn}')
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    acc = (tp +tn)/cm.sum()
    specificity = tn/(fp+tn)
    npv = tn/(tn+fn)
    f1 = stats.harmonic_mean([precision,recall])
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'Accuracy: {acc}')
    print(f'Specificity: {specificity}')
    print(f'NPV: {npv}')
    print(f'F1 score: {f1}')
# model1
# TP: 8   FP: 2
# FN: 8   TN: 182
# Recall: 0.5
# Precision: 0.8
# Accuracy: 0.95
# Specificity: 0.9891304347826086
# NPV: 0.9578947368421052
# F1 score: 0.6153846153846154

# model2
# TP: 9   FP: 81
# FN: 7   TN: 103
# Recall: 0.5625
# Precision: 0.1
# Accuracy: 0.56
# Specificity: 0.5597826086956522
# NPV: 0.9363636363636364
# F1 score: 0.16981132075471697

# model3
# TP: 13   FP: 86
# FN: 3   TN: 98
# Recall: 0.8125
# Precision: 0.13131313131313133
# Accuracy: 0.555
# Specificity: 0.532608695652174
# NPV: 0.9702970297029703
# F1 score: 0.22608695652173913

#3b) if the goal is to minimize vacation packages for non-defective ducks, 
# then we want to minimize False Negative 

# ANS: Both times we want to minimize false negatives and therefore have  high recall.
#  MODEL 3 has the highest recall by far and should be used

#4)
cd = pd.read_csv('https://ds.codeup.com/data/gives_you_paws.csv')
#get the most common actual and make new baseline column with only that
cd['baseline'] = cd.actual.mode()[0] 

for i in cd.columns[1:]:
    print(f'\n\033[1m{i}\033[0m')
    print(classification_report(cd.actual,cd[i],zero_division=0))
#USING output below:
# 4a) Model 1, then model 4 did the best at identifying dogs. Both of which did better than the baseline
# 4b) for dog team, I would choose a model with high precision with dog being positive 
#      in order to reduce the number of false positives.  I would choose model 1 for phase I then.
#     I would also choose it for phase II as it is the best model anyway
# 4c) for the cat team, I would choose a model with high precision w/ cat being target
#     The model with the highest precision for cats is model 4, so I would choose that for phase I
#     However, model 4 is less accurate than model 1, so I would choose model 1 for phase II

# model1
#               precision    recall  f1-score   support

#          cat       0.69      0.82      0.75      1746
#          dog       0.89      0.80      0.84      3254

#     accuracy                           0.81      5000
#    macro avg       0.79      0.81      0.80      5000
# weighted avg       0.82      0.81      0.81      5000


# model2
#               precision    recall  f1-score   support

#          cat       0.48      0.89      0.63      1746
#          dog       0.89      0.49      0.63      3254

#     accuracy                           0.63      5000
#    macro avg       0.69      0.69      0.63      5000
# weighted avg       0.75      0.63      0.63      5000


# model3
#               precision    recall  f1-score   support

#          cat       0.36      0.51      0.42      1746
#          dog       0.66      0.51      0.57      3254

#     accuracy                           0.51      5000
#    macro avg       0.51      0.51      0.50      5000
# weighted avg       0.55      0.51      0.52      5000


# model4
#               precision    recall  f1-score   support

#          cat       0.81      0.35      0.48      1746
#          dog       0.73      0.96      0.83      3254

#     accuracy                           0.74      5000
#    macro avg       0.77      0.65      0.66      5000
# weighted avg       0.76      0.74      0.71      5000


# baseline
#               precision    recall  f1-score   support

#          cat       0.00      0.00      0.00      1746
#          dog       0.65      1.00      0.79      3254

#     accuracy                           0.65      5000
#    macro avg       0.33      0.50      0.39      5000
# weighted avg       0.42      0.65      0.51      5000

#5) 
# classification_report already used above.
for i in cd.columns[1:]:
    print(f'\n\033[1m{i}\033[0m')
    print(f'accuracy: {accuracy_score(cd.actual,cd[i])}')
    p=precision_score(cd.actual,cd[i],pos_label='dog')
    r=recall_score(cd.actual,cd[i],pos_label='dog')
    print(f'precision: {p}')
    print(f'recall: {r}')
# model1
# accuracy: 0.8074
# precision: 0.8900238338440586
# recall: 0.803318992009834

# model2
# accuracy: 0.6304
# precision: 0.8931767337807607
# recall: 0.49078057775046097

# model3
# accuracy: 0.5096
# precision: 0.6598883572567783
# recall: 0.5086047940995697

# model4
# accuracy: 0.7426
# precision: 0.7312485304490948
# recall: 0.9557467732022127

# baseline
# accuracy: 0.6508
# precision: 0.6508
# recall: 1.0