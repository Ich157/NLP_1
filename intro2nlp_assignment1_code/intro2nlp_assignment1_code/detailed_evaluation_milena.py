# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import pandas as pd

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/spanish_model/model_output.tsv",sep='\t',header=None)
df = df.dropna()
label = df.iloc[:, 1].to_numpy()
predictions = df.iloc[:, 2].to_numpy()
print("recall:", recall_score(label, predictions, average=None))
print("precision", precision_score(label, predictions, average=None))
print("f1:", f1_score(label,predictions,average=None))
print("f1 weighted:", f1_score(label,predictions,average="weighted"))

print(classification_report(label, predictions))

#experimental results:
num_epochs = [2, 3, 5, 10, 20, 35, 50]
weightedF1= [0.99, 0.99, 0.99, 0.99, 0.985, 0.985, 0.985]

ax = plt.plot(num_epochs, weightedF1)
ax = plt.scatter(num_epochs, weightedF1)
plt.xlabel('num_epochs')
plt.ylabel('weighted F1 score')
plt.show()
