# step 1: import libraries
import time
import nltk
from nltk import FreqDist, pos_tag
from nltk.tokenize import word_tokenize
import matplotlib
import pandas as pd
from tabulate import tabulate

# step 2: download resources
def installations():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("tagsets", quiet=True)


# step 3: read input text
def getInput():
    text = input("Enter Input Text: ")
    return text


# step 4: perform word tokenization
def tokenizeText(text):
    tokens = word_tokenize(text)
    return tokens


# Step 5 & 6: perform pos tagging using nltk tagset definitions
def performPOSTagging(tokens):
    # Built-in NLTK pos_tag
    pos_tags = pos_tag(tokens)

    # penn treebank documentation dictionary
    tag_help_dict = nltk.data.load("help/tagsets/upenn_tagset.pickle")

    tagged_data = []
    for word, tag in pos_tags:
       # fetch desc from lib
        if tag in tag_help_dict:
            desc = tag_help_dict[tag][0]
        else:
            desc = "Punctuation / Symbol"
        tagged_data.append((word, tag, desc))

    df_pos = pd.DataFrame(
        tagged_data, columns=["Word", "POS Tag", "Description"]
    )
    return pos_tags, df_pos


# step 7: count pos tag freq
def countPOSFrequencies(pos_tags):
    tags_only = [tag for _, tag in pos_tags]
    tag_freq = FreqDist(tags_only)

    df_freq = pd.DataFrame(
        tag_freq.most_common(), columns=["POS Tag", "Frequency"]
    )
    return df_freq


# step 8: analyze broad linguistic categories
def analyzeResults(pos_tags):
    nouns = sum(1 for _, tag in pos_tags if tag.startswith("NN"))
    verbs = sum(1 for _, tag in pos_tags if tag.startswith("VB"))
    adjectives = sum(1 for _, tag in pos_tags if tag.startswith("JJ"))
    adverbs = sum(1 for _, tag in pos_tags if tag.startswith("RB"))

    print("\n--- STEP 8: ANALYSIS ---")
    print(f"1. Total Nouns (NN, NNS, NNP, NNPS): {nouns}")
    print(f"2. Total Verbs (VB, VBD, VBG, VBN, VBP, VBZ): {verbs}")
    print(f"3. Total Adjectives (JJ, JJR, JJS): {adjectives}")
    print(f"4. Total Adverbs (RB, RBR, RBS, WRB): {adverbs}")


def main():
    print("Performing Installations...")
    installations()
    time.sleep(1)
    print("Installation Successful!\n")

    # Step 3
    orig_text = getInput()
    time.sleep(1)
    print("\n--- ORIGINAL TEXT ---")
    print(orig_text)

    # Step 4
    tokens = tokenizeText(orig_text)
    time.sleep(1)
    print("\n--- WORD TOKENS ---")
    print(tokens)

    # Step 5 & 6
    pos_tags, df_pos = performPOSTagging(tokens)
    time.sleep(1)
    print("\n--- POS TAGS TABLE ---")
    print(tabulate(df_pos, headers="keys", tablefmt="psql", showindex=False))

    # Step 7
    df_freq = countPOSFrequencies(pos_tags)
    time.sleep(1)
    print("\n--- POS TAG FREQUENCY TABLE ---")
    print(tabulate(df_freq, headers="keys", tablefmt="psql", showindex=False))

    # Step 8
    time.sleep(1)
    analyzeResults(pos_tags)


if __name__ == "__main__":
    main()