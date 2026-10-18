# step 1: imports
import math
import re
import time
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from tabulate import tabulate


# step 2: installations
def installations():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


# step 3: corpus
def getCorpus():
    documents = [
        "Machine learning is used for machine fault detection.",
        "Machine learning helps engineers detect faults.",
        "Deep learning is used for image classification.",
    ]
    return documents


# step 4: preprocessing
def preprocessDocument(doc):
    # lowercase text
    lower_text = doc.lower()

    # remove punctuation
    clean_text = re.sub(r"[^\w\s]", "", lower_text)

    # word tokenization
    try:
        tokens = word_tokenize(clean_text)
    except Exception:
        tokens = clean_text.split()
    return tokens


# step 5: tf
def calculateTF(tokenized_docs, vocab):
    tf_list = []
    for tokens in tokenized_docs:
        total_tokens = len(tokens)
        counts = Counter(tokens)
        doc_tf = {word: counts[word] / total_tokens for word in vocab}
        tf_list.append(doc_tf)
    return tf_list


# step 6: idf
def calculateIDF(tokenized_docs, vocab):
    total_docs = len(tokenized_docs)
    idf_dict = {}
    for word in vocab:
        doc_freq = sum(1 for tokens in tokenized_docs if word in tokens)
        idf_dict[word] = math.log10(total_docs / doc_freq)
    return idf_dict


# step 7: tfidf
def calculateTFIDF(tf_list, idf_dict, vocab):
    tfidf_list = []
    for doc_tf in tf_list:
        doc_tfidf = {
            word: round(doc_tf[word] * idf_dict[word], 4) for word in vocab
        }
        tfidf_list.append(doc_tfidf)
    return tfidf_list


# step 8: tables
def displayMatrices(tokenized_docs, vocab, tf_list, idf_dict, tfidf_list):
    doc_labels = [f"Document {i+1}" for i in range(len(tokenized_docs))]

    # idf table
    df_idf = pd.DataFrame(
        [
            {
                "Word": word,
                "Doc Frequency": sum(
                    1 for tokens in tokenized_docs if word in tokens
                ),
                "IDF": round(idf_dict[word], 4),
            }
            for word in vocab
        ]
    )
    print("\n--- IDF VALUES ---")
    print(tabulate(df_idf, headers="keys", tablefmt="psql", showindex=False))

    # tfidf table
    tfidf_matrix = []
    for i, doc_tfidf in enumerate(tfidf_list):
        row = [doc_labels[i]] + [doc_tfidf[word] for word in vocab]
        tfidf_matrix.append(row)

    df_tfidf = pd.DataFrame(tfidf_matrix, columns=["Document"] + vocab)
    print("\n--- TF-IDF MATRIX ---")
    print(tabulate(df_tfidf, headers="keys", tablefmt="psql", showindex=False))

    return df_tfidf


# step 9: analysis
def analyzeHighTFIDF(tfidf_list, doc_labels, vocab):
    print("\n--- HIGH TF-IDF WORDS ---")
    for i, doc_tfidf in enumerate(tfidf_list):
        sorted_words = sorted(
            doc_tfidf.items(), key=lambda item: item[1], reverse=True
        )
        max_val = sorted_words[0][1]
        top_words = [
            (word, score) for word, score in sorted_words if score == max_val
        ]

        print(f"\n{doc_labels[i]}:")
        print(f"   - Highest Score: {max_val}")
        print(
            f"   - Top Word(s): {', '.join(word for word, _ in top_words)}"
        )


# main function
def main():
    print("Performing Installations...")
    installations()
    time.sleep(1)
    print("Installation Successful!\n")

    # get corpus
    documents = getCorpus()
    print("--- ORIGINAL DOCUMENTS ---")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {doc}")

    # perform preprocessing
    time.sleep(1)
    print("\n--- PREPROCESSED TOKENS ---")
    tokenized_docs = []
    for i, doc in enumerate(documents):
        tokens = preprocessDocument(doc)
        tokenized_docs.append(tokens)
        print(f"Document {i+1}: {tokens}")

    # build vocabulary
    vocab = sorted(list({word for tokens in tokenized_docs for word in tokens}))

    # calculate representations
    tf_list = calculateTF(tokenized_docs, vocab)
    idf_dict = calculateIDF(tokenized_docs, vocab)
    tfidf_list = calculateTFIDF(tf_list, idf_dict, vocab)

    # display results
    time.sleep(1)
    displayMatrices(tokenized_docs, vocab, tf_list, idf_dict, tfidf_list)

    # analyze keywords
    time.sleep(1)
    doc_labels = [f"Document {i+1}" for i in range(len(documents))]
    analyzeHighTFIDF(tfidf_list, doc_labels, vocab)


# run program
if __name__ == "__main__":
    main()
