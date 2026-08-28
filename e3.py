import time
import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import pandas as pd
from tabulate import tabulate


# step 1 & 2: installations and imports
def installations():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# step 3: read cleaned input text
def getInput():
    text = input("Enter Input Text: ")
    # cleaning text: lowercase and remove non-alphanumeric characters
    cleaned = "".join(
        char.lower() for char in text if char.isalnum() or char.isspace()
    )
    return text, cleaned


# step 4: perform word tokenization
def tokenizeText(cleaned_text):
    tokens = word_tokenize(cleaned_text)
    return tokens


# step 5: generate n-grams and calculate frequencies
def generateNGrams(tokens):
    # unigrams
    unigrams_list = list(ngrams(tokens, 1))
    unigram_freq = FreqDist(tokens)

    # bigrams
    bigrams_list = list(ngrams(tokens, 2))
    bigram_freq = FreqDist(bigrams_list)

    # trigrams
    trigrams_list = list(ngrams(tokens, 3))
    trigram_freq = FreqDist(trigrams_list)

    return (
        unigrams_list,
        unigram_freq,
        bigrams_list,
        bigram_freq,
        trigrams_list,
        trigram_freq,
    )


# step 6: create separate tables for each n-gram
def displayTables(unigram_freq, bigram_freq, trigram_freq):
    # unigram table
    df_uni = pd.DataFrame(
        unigram_freq.most_common(), columns=["Unigram", "Frequency"]
    )
    print("\n--- UNIGRAM TABLE ---")
    print(tabulate(df_uni, headers="keys", tablefmt="psql", showindex=False))

    # bigram table
    df_bi = pd.DataFrame(
        [(" ".join(k), v) for k, v in bigram_freq.most_common()],
        columns=["Bigram", "Frequency"],
    )
    print("\n--- BIGRAM TABLE ---")
    print(tabulate(df_bi, headers="keys", tablefmt="psql", showindex=False))

    # trigram table
    df_tri = pd.DataFrame(
        [(" ".join(k), v) for k, v in trigram_freq.most_common()],
        columns=["Trigram", "Frequency"],
    )
    print("\n--- TRIGRAM TABLE ---")
    print(tabulate(df_tri, headers="keys", tablefmt="psql", showindex=False))

    return df_uni, df_bi, df_tri


# step 7: analyze the results
def analysis(df_uni, df_bi, df_tri):
    print("\n--- ANALYSIS ---")

    if not df_uni.empty:
        print("1. Most frequent word in the text:")
        print(
            f"   - '{df_uni.iloc[0]['Unigram']}' (Frequency: {df_uni.iloc[0]['Frequency']})"
        )
    else:
        print("1. Most frequent word: None")

    if not df_bi.empty:
        print("2. Most frequently occurring word pair:")
        print(
            f"   - '{df_bi.iloc[0]['Bigram']}' (Frequency: {df_bi.iloc[0]['Frequency']})"
        )
    else:
        print("2. Most frequently occurring word pair: None")

    if not df_tri.empty:
        print("3. Most frequently occurring three-word sequence:")
        print(
            f"   - '{df_tri.iloc[0]['Trigram']}' (Frequency: {df_tri.iloc[0]['Frequency']})"
        )
    else:
        print("3. Most frequently occurring three-word sequence: None")


def main():
    print("Performing Installations...")
    installations()
    time.sleep(1)
    print("Installation Successful!\n")

    orig_text, cleaned_text = getInput()
    time.sleep(1)
    print("\nOriginal Text:\n", orig_text)
    print("\nCleaned Text:\n", cleaned_text)

    tokens = tokenizeText(cleaned_text)
    time.sleep(1)
    print("\nWord Tokens:")
    print(tokens)

    (
        unigrams_list,
        unigram_freq,
        bigrams_list,
        bigram_freq,
        trigrams_list,
        trigram_freq,
    ) = generateNGrams(tokens)

    time.sleep(1)
    df_uni, df_bi, df_tri = displayTables(
        unigram_freq, bigram_freq, trigram_freq
    )

    time.sleep(1)
    analysis(df_uni, df_bi, df_tri)


if __name__ == "__main__":
    main()