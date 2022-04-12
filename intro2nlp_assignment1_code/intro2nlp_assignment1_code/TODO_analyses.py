# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
#from intro2nlp_assignment1_code.intro2nlp_assignment1_code.train import train
import spacy

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
typelist = {}
for line in train_sentences:
    line = nlp(line)
    for token in line:
        tokensize+=1
        if token in typelist.keys():
            typelist[token] += 1
        if token not in typelist:
            typelist[token] = 1
        #print(token.text, token.pos_, token.dep_)

print("Token size:", tokensize) # 16130
print("Type size:", typelist) # 16130
#print(typelist)

### WORD CLASSES ###

### N-GRAMS ###

### LEMMATIZATION ###

### NAMED ENTITY RECOGNITION ###