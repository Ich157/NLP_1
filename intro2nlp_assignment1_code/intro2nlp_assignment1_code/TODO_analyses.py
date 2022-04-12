# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
#from intro2nlp_assignment1_code.intro2nlp_assignment1_code.train import train
import spacy
import statistics

# import dataset
if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/dev/"
    test_path = "data/preprocessed/test/"

    with open(train_path + "sentences.txt", encoding="utf8") as sent_file:
        train_sentences = sent_file.readlines()

### TOKENIZATION ###
nlp = spacy.load("en_core_web_sm")
tokensize = 0
typelist = []
words = []
no_words = []
for line in train_sentences:
    line = nlp(line)
    linelength = 0
    for token in line:
        tokensize+=1
        if token.text not in typelist:
            typelist.append(token.text)
        # Let's filter out punctuation
        if not token.is_punct:
            words.append(token.text)
            linelength +=1
        #print(token.text, token.pos_, token.dep_)
    no_words.append(linelength)

print("Token size:", tokensize) # 16130
print("Type size:", len(typelist)) # 3746
print("Number of words", len(words)) # 13895
print("Average number of words", round(statistics.mean(no_words), 2)

### WORD CLASSES ###

### N-GRAMS ###

### LEMMATIZATION ###

### NAMED ENTITY RECOGNITION ###
