### QUESTION 8 ###
import spacy
import pandas as pd
from wordfreq import word_frequency as wf
import matplotlib.pyplot as plt
data = pd.read_csv("data/original/spanish/Spanish_Train.tsv", sep='\t', usecols=[1,4,9,10], header=None)
data.columns = ["sentence","target","binary_label", "probabilistic label"]
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

#### 8
data2 = data[data.binary_label == 1]
list = []
list_c = []
list_feq = []
list_pos = []
#get length of target words and word count of target
for target in data2["target"]:
    target = nlp(target)
    list.append(len(target.doc))
    list_c.append(len(target.text))
data2["word_count"] = list
data2["char_count"] = list_c
#get rid of all lines that have a target thats contains more than one word
data2 = data2[data2.word_count == 1]

#iterate over dataframe to get the pos of every traget
for index, row in data2.iterrows():
    line = nlp(row.sentence)
    for token in line:
        if token.text == row.target:
            list_pos.append(token.pos_)
            break
last_row = data2.iloc[-1]

line = nlp(last_row.sentence)
list_pos.extend(token.pos_ for token in line if token.text == last_row.target)
data2["pos"] = list_pos
print(data2)
for target in data2["target"]:
    feq = wf(target, "en")
    list_feq.append(feq)
data2["frequency"] = list_feq
print("pearson correlation length and complexity")
print(data2["char_count"].corr(data2["probabilistic label"], method="pearson"))
print("frequency and complexity")
print(data2["frequency"].corr(data2["probabilistic label"], method="pearson"))

#data2.plot.scatter(x = 'char_count', y = 'probabilistic label', s = 100)
#data2.plot.scatter(x = 'frequency', y = 'probabilistic label', s = 100)

plt.scatter(x=data2["char_count"], y=data2["probabilistic label"])
plt.ylabel("complexity")
plt.xlabel("Length")
plt.title("Scatterplot of word complexity and word length")
plt.show()
plt.ylabel("complexity")
plt.xlabel("frequency")
plt.title("Scatterplot of word complexity and frequency of word")
plt.scatter(x=data2["frequency"], y=data2["probabilistic label"])
plt.show()
plt.ylabel("complexity")
plt.xlabel("POS")
plt.title("Scatterplot of complexity of word and POS label")
plt.scatter(x=data2["pos"], y=data2["probabilistic label"])
plt.show()