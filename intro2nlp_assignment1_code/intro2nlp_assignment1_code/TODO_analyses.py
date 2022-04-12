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

    # open
    with open(train_path + "sentences.txt", encoding="utf8") as sent_file:
        train_sentences = sent_file.readlines()

# set nlp command
nlp = spacy.load("en_core_web_sm")

question = 3

if question == 1:
    ### 1. TOKENIZATION ###
    tokensize = 0
    typelist = []
    words = []
    no_words = []
    words_length = []
    for line in train_sentences:
        # tokenize line
        line = nlp(line)
        linelength = 0
        wordlength_line = []
        for token in line:
            # tokens are counted by just adding 1 for every token
            tokensize+=1
            # save unique tokens as types in a list
            if token.text not in typelist:
                typelist.append(token.text)
            # filter out punctuation, save only words
            if not token.is_punct:
                words.append(token.text)
                linelength +=1
                wordlength = sum(1 for _ in token.text)
                wordlength_line.append(wordlength)
        no_words.append(linelength)
        words_length.append(statistics.mean(wordlength_line))

    print("Token size:", tokensize) # 16130
    # number of tokens is the length of the list of types
    print("Type size:", len(typelist)) # 3746
    print("Number of words", len(words)) # 13895
    avg = round(statistics.mean(no_words), 2)
    print("Average number of words", avg) # 21.28
    avglen = round(statistics.mean(words_length), 2)
    print("Average word length", avglen) # 4.66

### 2. WORD CLASSES ###
elif question == 2:
    pos_freqs = {}
    tag_freqs = {}
    fine_to_uni = {}
    tag_token_freqs = {}
    alltokens = 0

    table = []

    for line in train_sentences:
        line = nlp(line)
        for token in line:
            # print(token.text, token.pos_, token.tag_)
            alltokens +=1
            # finegrained tag
            if token.tag_ in tag_freqs:
                tag_freqs[token.tag_] += 1
                fine_to_uni[token.tag_] = token.pos_
            else:
                tag_freqs[token.tag_] = 1
            if (token.tag_, token.text) in tag_token_freqs:
                tag_token_freqs[(token.tag_, token.text)] += 1
            else:
                tag_token_freqs[(token.tag_, token.text)] = 1
            if token.pos_ in pos_freqs:
                pos_freqs[token.pos_] += 1
            else:
                pos_freqs[token.pos_] = 1
    pos_sorted = sorted(pos_freqs, key=pos_freqs.get, reverse=True)
    tag_sorted = sorted(tag_freqs, key=tag_freqs.get, reverse=True)
    tag_token_sorted = sorted(tag_token_freqs, key=tag_token_freqs.get, reverse = True)
    tag_token_sorted_ascending = sorted(tag_token_freqs, key=tag_token_freqs.get, reverse = False)

    for n in range(10):
        frequenttag = tag_sorted[n]
        frequency = tag_freqs[frequenttag]
        unilabel = fine_to_uni[frequenttag]
        relative = round(frequency / alltokens, 2)
        frequenttokens = []
        infrequenttoken = 0
        #print(tag_token_sorted)

        for entry in tag_token_sorted:
            if len(frequenttokens) < 3 and entry[0] == frequenttag:
                frequenttokens.append(entry[1])
            elif infrequenttoken == 0 and entry[0] == frequenttag and tag_token_freqs[entry] == 1:
                infrequenttoken = entry[1]
        if infrequenttoken == 0:
            # select first entry you can find, unless it is frequent
            for entry in tag_token_sorted_ascending:
                if infrequenttoken == 0 and entry[0] == frequenttag and tag_token_freqs[entry] == 1 and entry[1] not in frequenttokens:
                    infrequenttoken = entry[1]

        info = [frequenttag, unilabel, frequency, f"{relative*100}%", frequenttokens, infrequenttoken]
        table.append(info)

    print(table)

### 3. N-GRAMS ###

### 4. LEMMATIZATION ###
lemma_freq = {}
lemma_tokens_lines = {}
lemmas = []

for line in train_sentences:
    line = nlp(line)
    for token in line:
        #print(token.text, token.lemma_)
        if token.lemma_ in lemma_freq:
            lemma_freq[token.lemma_] += 1
            lemma_tokens_lines[token.lemma_] = lemma_tokens_lines[token.lemma_] + (token.text, line)
        else:
            lemma_freq[token.lemma_] = 1
            lemmas.append(token.lemma_)
            lemma_tokens_lines[token.lemma_] = (token.text, line)

frequent_lemma = 0
inflections = []
lines = []

for lemma in lemmas:
    if lemma_freq[lemma] > 2 and frequent_lemma == 0:
        appearances = lemma_tokens_lines[lemma]
        inflections.append(appearances[0])
        inflections.append(appearances[1])
        lines.append(appearances[2])
        lines.append(appearances[3])

        frequent_lemma = (lemma, inflections, lines)

print(frequent_lemma)

### 5. NAMED ENTITY RECOGNITION ###