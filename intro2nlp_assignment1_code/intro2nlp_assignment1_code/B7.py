import spacy
import pandas as pd
from wordfreq import word_frequency as wf
data = pd.read_csv("data/original/english/WikiNews_Train.tsv", sep='\t', usecols=[4,9,10], header=None)
data.columns = ["target","binary_label", "probabilistic label"]

print("binary_label")
print(data["binary_label"].value_counts())

print("max probabilistic label")
print(max(data["probabilistic label"]))
print("min probabilistic label")
print(min(data["probabilistic label"]))
print("mean probabilistic label")
print(data["probabilistic label"].mean())
print("median probabilistic label")
print(data["probabilistic label"].median())
print("stdev probabilistic label")
print(data["probabilistic label"].std())

nlp = spacy.load("en_core_web_sm")
#counter = 0
#max = 0
#for target in data["target"]:
#    target = nlp(target)
#    if(len(target.doc)>1):
#        counter = counter + 1
#    if(len(target.doc)>max):
#        max = len(target.doc)
#        print(target.doc)
#print(counter)
#print(max)

#### 8
data2 = data[data.binary_label == 1]
list = []
list_c = []
list_feq = []
for target in data2["target"]:
    target = nlp(target)
    list.append(len(target.doc))
    list_c.append(len(target.text))
data2["word_count"] = list
data2["char_count"] = list_c
data2 = data2[data2.word_count == 1]
for target in data2["target"]:
    feq = wf(target, "en")
    list_feq.append(feq)
data2["frequency"] = list_feq
print("pearson correlation length and complexity")
print(data2["char_count"].corr(data2["probabilistic label"], method="pearson"))
print("frequency and complexity")
print(data2["frequency"].corr(data2["probabilistic label"], method="pearson"))

data2.plot.scatter(x = 'char_count', y = 'probabilistic label', s = 100)
data2.plot.scatter(x = 'frequency', y = 'probabilistic label', s = 100)

