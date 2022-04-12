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

### 1. TOKENIZATION ###
nlp = spacy.load("en_core_web_sm")
tokensize = 0
typelist = []
words = []
no_words = []
words_length = []
for line in train_sentences:
    line = nlp(line)
    linelength = 0
    wordlength_line = []
    for token in line:
        tokensize+=1
        if token.text not in typelist:
            typelist.append(token.text)
        # Let's filter out punctuation
        if not token.is_punct:
            words.append(token.text)
            linelength +=1
            wordlength = 0
            for letter in token.text:
                wordlength += 1
            wordlength_line.append(wordlength)
        #print(token.text, token.pos_, token.dep_)
    no_words.append(linelength)
    words_length.append(statistics.mean(wordlength_line))

print("Token size:", tokensize) # 16130
print("Type size:", len(typelist)) # 3746
print("Number of words", len(words)) # 13895
avg = round(statistics.mean(no_words), 2)
print("Average number of words", avg) # 21.28
avglen = round(statistics.mean(words_length), 2)
print("Average word length", avglen) # 4.66

### 2. WORD CLASSES ###
pos_freqs = {}
tag_freqs = {}

table = []

for line in train_sentences:
    line = nlp(line)
    for token in line:
        # print(token.text, token.pos_, token.tag_)
        if token.pos_ in pos_freqs:
            pos_freqs[token.pos_] += 1

        else:
            pos_freqs[token.pos_] = 1
        if token.tag_ in tag_freqs:
            tag_freqs[token.tag_] += 1
        else:
            tag_freqs[token.tag_] = 1
pos_sorted = sorted(pos_freqs, key=pos_freqs.get, reverse=True)
print(pos_sorted)
print(pos_freqs)
tag_sorted = sorted(tag_freqs, key=tag_freqs.get, reverse=True)
print(tag_sorted)
print(tag_freqs)

for n in range(10):
    frequenttag = tag_sorted[n]
    frequency = tag_freqs[frequenttag]
    info = [frequenttag, frequency]
table.append(info)

### 3. N-GRAMS ###

### 4. LEMMATIZATION ###

### 5. NAMED ENTITY RECOGNITION ###