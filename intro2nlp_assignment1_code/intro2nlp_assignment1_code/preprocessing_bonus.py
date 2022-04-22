import pandas as pd
import spacy

dev_data = pd.read_csv('data/original/german/German_Train.tsv', sep='\t', names=["HIT","sentence","start","end","target","native","non-native","marked_native","makred_non-native","binary","probabilistic"])
dev_sentences = []
dev_labels = []
for index, row in dev_data.iterrows():
    dev_sentences.append(row["sentence"])
dev_sentences = set(dev_sentences)
nlp = spacy.load("de_core_news_sm")
labels_file = open("data/preprocessed/german/train/labels.txt","w")
sentences_file = open("data/preprocessed/german/train/sentences.txt","w")
for sentence in dev_sentences:
    sentence = nlp(sentence)
    sentences_file.write(sentence.text)
    sentences_file.write("\n")
    for token in sentence:
        if token.text in dev_data["target"].values:
            df_row = dev_data.loc[dev_data["sentence"] == sentence.text]
            df_row = df_row.loc[df_row["target"] == token.text]
            #try:
            if len(df_row)>1:
                df_row = df_row.iloc[0]
            if df_row["binary"].any() == 1:
                labels_file.write("C ")
            else:
                labels_file.write("N ")
            #except:
            #    print(df_row["binary"])
        else:
            labels_file.write("N ")
    labels_file.write("\n")
labels_file.close()

