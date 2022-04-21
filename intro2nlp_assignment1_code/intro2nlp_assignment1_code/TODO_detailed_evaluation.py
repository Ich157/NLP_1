# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import pandas as pd

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

df = pd.read_csv("experiments/base_model/model_output.tsv",sep='\t',header=None)
df = df.dropna()
label = df.iloc[:, 1].to_numpy()
predictions = df.iloc[:, 2].to_numpy()
print(recall_score(label, predictions, average=None))
print(precision_score(label, predictions, average=None))
print(f1_score(label,predictions,average=None))
print(f1_score(label,predictions,average="weighted"))

print(classification_report(label, predictions))