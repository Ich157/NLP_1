# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
#from intro2nlp_assignment1_code.intro2nlp_assignment1_code.train import train
import spacy
import statistics
from collections import Counter

# import dataset
if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/dev/"
    test_path = "data/preprocessed/test/"

    # open file
    with open(train_path + "sentences.txt", encoding="utf8") as sent_file:
        train_sentences = sent_file.readlines()

# set nlp command
nlp = spacy.load("en_core_web_sm")

# you can adjust the question number here
question = 4

if question == 1:
    ### 1. TOKENIZATION ###
    # create lists & counter for variables
    no_words = []
    words_length = []

    word_frequencies = Counter()
    num_tokens = []
    for line in train_sentences:
        # tokenize line
        line = nlp(line)
        linelength = 0
        wordlength_line = []
        words = []
        for token in line:
            # tokens are counted by just adding 1 for every token
            num_tokens.append(token)
            # filter out punctuation, save only words
            if not token.is_punct:
                words.append(token.text)
                linelength +=1
                # calculate the length of the word
                wordlength = sum(1 for _ in token.text)
                # save in list for all words in a sentence
                words_length.append(wordlength)
        # save frequencies of word and number of words in a line
        word_frequencies.update(words)
        no_words.append(linelength)

    num_tokens = len(num_tokens)
    num_words = sum(word_frequencies.values())
    num_types = len(word_frequencies.keys())
    avg = round(statistics.mean(no_words), 2)
    avglen = round(statistics.mean(words_length), 2)

    print("Token size:", num_tokens) # 16130
    print("Type size:", num_types) # 3722
    print("Number of words:", num_words) # 13895
    print("Average number of words:", avg) # 21.28
    print("Average word length:", avglen) # 4.72

### 2. WORD CLASSES ###
if question == 2:
    # create variables to save data
    pos_freqs = {}
    tag_freqs = {}
    # translation of finegrained tag to universal tag
    fine_to_uni = {}
    # frequencies of tag-token combos
    tag_token_freqs = {}
    alltokens = 0

    table = []

    for line in train_sentences:
        line = nlp(line)
        for token in line:
            alltokens +=1
            # finegrained tag
            if token.tag_ in tag_freqs:
                tag_freqs[token.tag_] += 1
                fine_to_uni[token.tag_] = token.pos_
            else:
                tag_freqs[token.tag_] = 1
            # also save combination of tag and token to find frequent/infrequent tags
            if (token.tag_, token.text) in tag_token_freqs:
                tag_token_freqs[(token.tag_, token.text)] += 1
            else:
                tag_token_freqs[(token.tag_, token.text)] = 1
            # universal tag
            if token.pos_ in pos_freqs:
                pos_freqs[token.pos_] += 1
            else:
                pos_freqs[token.pos_] = 1
    # sort by size to find most/least frequent tags
    pos_sorted = sorted(pos_freqs, key=pos_freqs.get, reverse=True)
    tag_sorted = sorted(tag_freqs, key=tag_freqs.get, reverse=True)
    tag_token_sorted = sorted(tag_token_freqs, key=tag_token_freqs.get, reverse = True)
    tag_token_sorted_ascending = sorted(tag_token_freqs, key=tag_token_freqs.get, reverse = False)

    for n in range(10):
        # only for 10 most frequent tokens
        # find tag name
        frequenttag = tag_sorted[n]
        # find frequency of this tag
        frequency = tag_freqs[frequenttag]
        # translate this (finegrained) label to a universal label
        unilabel = fine_to_uni[frequenttag]
        # find relative frequency of tag
        relative = round(frequency / alltokens, 2)
        frequenttokens = []
        infrequenttoken = 0

        for entry in tag_token_sorted:
            # find 3 most frequent tokens with this tag
            if len(frequenttokens) < 3 and entry[0] == frequenttag:
                frequenttokens.append(entry[1])
            # choose a random infrequent token (frequency of 0)
            elif infrequenttoken == 0 and entry[0] == frequenttag and tag_token_freqs[entry] == 1:
                infrequenttoken = entry[1]
        # some tags don't really have infrequent tags. For this, we sort the tags in ascending order and select the least common tag
        # however, we make sure that the tag is not one of the three most frequent ones
        if infrequenttoken == 0:
            # select first entry you can find, unless it is frequent
            for entry in tag_token_sorted_ascending:
                if infrequenttoken == 0 and entry[0] == frequenttag and tag_token_freqs[entry] == 1 and entry[1] not in frequenttokens:
                    infrequenttoken = entry[1]

        # combine all information
        info = [frequenttag, unilabel, frequency, f"{relative*100}%", frequenttokens, infrequenttoken]
        table.append(info)

    # print all combined information
    print(table)

### 3. N-GRAMS ###
elif question == 3:
    # create variables
    new_token = 0
    bigrams = {}
    new_pos = 0
    pos_bigrams = {}

    for line in train_sentences:
        line = nlp(line)
        for token in line:
            # make the previously saved token the old token
            prev_token = new_token
            # save the current token as new token
            new_token = token.text
            # if there is a previous token:
            if not prev_token == 0:
                # the bigram consists of the old and new token
                bigram = (prev_token, new_token)
                # add frequency to dictionary
                if bigram in bigrams:
                    bigrams[bigram] += 1
                else:
                    bigrams[bigram] = 1
            # repeat for POS bigrams
            prev_pos = new_pos
            new_pos = token.pos_
            if not prev_pos == 0:
                pos_bigram = (prev_pos, new_pos)
                if pos_bigram in pos_bigrams:
                    pos_bigrams[pos_bigram] += 1
                else:
                    pos_bigrams[pos_bigram] = 1

    # sort and print most common bigrams/POS bigrams
    bigrams_sorted = sorted(bigrams, key=bigrams.get, reverse=True)
    print("Most frequent bigrams:", bigrams_sorted[0:3])

    pos_bigrams_sorted = sorted(pos_bigrams, key=pos_bigrams.get, reverse=True)
    print("Most frequent POS bigrams:", pos_bigrams_sorted[0:3])

    # repeat all for the trigrams.
    new_token = 0
    # the only difference is trigrams have a middle token
    middle_token = 0
    trigrams = {}
    new_pos = 0
    middle_pos = 0
    pos_trigrams = {}

    for line in train_sentences:
        line = nlp(line)
        for token in line:
            first_token = middle_token
            middle_token = new_token
            new_token = token.text
            if not prev_token == 0:
                trigram = (first_token, middle_token, new_token)
                if trigram in trigrams:
                    trigrams[trigram] += 1
                else:
                    trigrams[trigram] = 1
            prev_pos = middle_pos
            middle_pos = new_pos
            new_pos = token.pos_
            if not prev_pos == 0:
                pos_trigram = (prev_pos, middle_pos, new_pos)
                if pos_trigram in pos_trigrams:
                    pos_trigrams[pos_trigram] += 1
                else:
                    pos_trigrams[pos_trigram] = 1

    trigrams_sorted = sorted(trigrams, key=trigrams.get, reverse=True)
    print("Most frequent trigrams:",trigrams_sorted[0:3])

    pos_trigrams_sorted = sorted(pos_trigrams, key=pos_trigrams.get, reverse=True)
    print("Most frequent POS trigrams:", pos_trigrams_sorted[0:3])

### 4. LEMMATIZATION ###
elif question == 4:
    # make variables
    lemma_freq = {}
    lemma_tokens = {}
    lemma_tokens_lines = {}
    lemmas = []

    for line in train_sentences:
        line = nlp(line)
        for token in line:
            # identify new lemmas, update dictionary
            if token.lemma_ in lemma_freq:
                # save frequency of each lemma
                lemma_freq[token.lemma_] += 1
                if token.text not in lemma_tokens[token.lemma_]:
                    # save lemma and all its inflections
                    lemma_tokens[token.lemma_].append(token.text)
                    # save lemma, inflection and line
                    lemma_tokens_lines[token.lemma_] = lemma_tokens_lines[token.lemma_] + (token.text, line)
            else:
                lemma_freq[token.lemma_] = 1
                lemmas.append(token.lemma_)
                lemma_tokens_lines[token.lemma_] = (token.text, line)
                lemma_tokens[token.lemma_] = [token.text]

    frequent_lemma = 0
    inflections = []
    lines = []

    for lemma in lemmas:
        # find lemmas with more than 2 inflections
        if len(lemma_tokens[lemma]) > 2 and frequent_lemma == 0:
            appearances = lemma_tokens_lines[lemma]
            inflections.append(appearances[0])
            inflections.append(appearances[1])
            lines.append(appearances[6])
            lines.append(appearances[7])

            frequent_lemma = (lemma, inflections, lines)

    print(frequent_lemma)

### 5. NAMED ENTITY RECOGNITION ###
elif question == 5:
    linecounter = 0
    entities = Counter()

    for line in train_sentences:
        line = nlp(line)
        line_entities = []
        for ent in line.ents:
            # save all entities
            line_entities.append(ent.text)
            if linecounter < 5:
                # print entities of first 5 lines
                print(line)
                print(ent.text, ent.label_)
        entities.update(line_entities)
        linecounter += 1

    # calculate number of entities and entity labels
    num_entities = sum(entities.values())
    num_labels = len(entities.keys())

    print("Number of entities:", num_entities) #1648
    print("Number of different entity labels:", num_labels) #893