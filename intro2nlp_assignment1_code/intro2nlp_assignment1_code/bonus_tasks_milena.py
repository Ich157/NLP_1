### QUESTION 8 ###
import spacy
import pandas as pd
from wordfreq import word_frequency as wf
import matplotlib.pyplot as plt
import csv
import spacy

question = 12

if question == 8:

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

### PREPROCESSING FOR 12-14 ###
inputpath = "data/original/spanish"
outputpath = "data/preprocessed/spanish"

# build path to file & open
trainfile = inputpath + "/Spanish_Train.tsv"
testfile = inputpath + "/Spanish_Test.tsv"
devfile = inputpath + "/Spanish_Dev.tsv"

with open(trainfile, encoding="utf-8") as file:
    traindata = csv.reader(file, delimiter="\t")

    nlp = spacy.load("es_core_news_sm")

    sentences = []
    labels = []

    # some sentences produce errors
    errorsentences = ["El , 21 de febrero y el 2 de septiembre de 2011, varias extensas áreas fueron epicentro de sismos de 7,0; 5,9; y 6,9 grados en la escala de Richter, aunque sin causar daños ni víctimas, pues se registraron a profundidades de 600 km; y los movimientos telúricos llegaron a sacudir edificios altos en varias provincias, incluyendo la ciudad de Buenos Aires.", "La sismicidad del área de Santiago del Estero es frecuente y de intensidad baja, y un silencio sísmico de terremotos medios a graves cada 40 años.", "El , sismo de 1817 de 7,0 Richter, con máximos daños reportados al centro y norte de la provincia, donde se desplomaron casas y se produjo agrietamiento del suelo, los temblores duraron alrededor de una semana.", "En algunas de las casas sobre esas fisuras, el terreno quedó cubierto de más de 1 dm de arena.", "Su nombre se encuentra en la Lista Real de Abidos y fragmentos del Canon Real de Turín"]
     
    # saving relevant data for each line
    for line in traindata:
        # label for goal word
        label = int(line[-2])
        # sentence
        sentence = line[1]
        # goal word
        word = line[4]

        sentence_nlp = nlp(sentence)
        index = 0
        wordindex = 0

        for word_nlp in sentence_nlp:
            if word_nlp.text == word:
                # save index of goal word
                wordindex = index
            index += 1


        if sentence not in sentences and sentence not in errorsentences:
            sentences.append(sentence)
            length_sentence = [0]*(len(sentence_nlp))
            labels.append(length_sentence)
        else:
            if label == 1 and sentence not in errorsentences:
                labels[-1][wordindex] = 1

# TODO
# save data as txt files
with open(outputpath + '/train/sentences.txt', 'w', encoding="utf-8") as f:
    for sentence in sentences:
        sentence = nlp(sentence)
        item = str( )
        for token in sentence:
            item = item + " " + token.text
        f.write("%s\n" % item)
with open(outputpath + '/train/labels.txt', 'w', encoding="utf-8") as f:
    for label in labels:
        item = str( )
        for number in label:
            item = item + " " + str(number)
        f.write("%s\n" % item)


# repeat for val data
with open(devfile, encoding="utf-8") as file:
    traindata = csv.reader(file, delimiter="\t")

    nlp = spacy.load("es_core_news_sm")

    sentences = []
    labels = []
     
    # saving relevant data for each line
    for line in traindata:
        # label for goal word
        label = int(line[-2])
        # sentence
        sentence = line[1]
        # goal word
        word = line[4]

        sentence_nlp = nlp(sentence)
        index = 0
        wordindex = 0

        for word_nlp in sentence_nlp:
            if word_nlp.text == word:
                # save index of goal word
                wordindex = index
            index += 1


        if sentence not in sentences:
            sentences.append(sentence)
            length_sentence = [0]*(len(sentence_nlp)-1)
            labels.append(length_sentence)
        else:
            if label == 1:
                labels[-1][wordindex] = 1

# TODO
# save data as txt files
with open(outputpath + '/val/sentences.txt', 'w', encoding="utf-8") as f:
    for sentence in sentences:
        sentence = nlp(sentence)
        item = str( )
        for token in sentence:
            item = item + " " + token.text
        f.write("%s\n" % item)
with open(outputpath + '/val/labels.txt', 'w', encoding="utf-8") as f:
    for label in labels:
        item = str( )
        for number in label:
            item = item + " " + str(number)
        f.write("%s\n" % item)

# repeat for test data
with open(testfile, encoding="utf-8") as file:
    traindata = csv.reader(file, delimiter="\t")

    nlp = spacy.load("es_core_news_sm")

    sentences = []
    labels = []
     
    # saving relevant data for each line
    for line in traindata:
        # label for goal word
        label = int(line[-2])
        # sentence
        sentence = line[1]
        # goal word
        word = line[4]

        sentence_nlp = nlp(sentence)
        index = 0
        wordindex = 0

        for word_nlp in sentence_nlp:
            if word_nlp.text == word:
                # save index of goal word
                wordindex = index
            index += 1


        if sentence not in sentences:
            sentences.append(sentence_nlp)
            length_sentence = [0]*(len(sentence_nlp)-1)
            labels.append(length_sentence)
        else:
            if label == 1 and sentence not in errorsentences:
                labels[-1][wordindex] = 1

# TODO
# save data as txt files
with open(outputpath + '/test/sentences.txt', 'w', encoding="utf-8") as f:
    for sentence in sentences:
        sentence = nlp(sentence)
        item = str( )
        for token in sentence:
            item = item + " " + token.text
        f.write("%s\n" % item)
with open(outputpath + '/test/labels.txt', 'w', encoding="utf-8") as f:
    for label in labels:
        item = str( )
        for number in label:
            item = item + " " + str(number)
        f.write("%s\n" % item)