import time
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from tabulate import tabulate


# step 1 & 2: installations and import
def installations():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


# step 3: corpus
def getCorpus():
    corpus = [
        [("The", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
        [("The", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
        [("A", "DET"), ("dog", "NOUN"), ("runs", "VERB")],
        [("A", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
    ]
    return corpus


# step 4: input
def getInput():
    try:
        text = input("Enter Input Text (press Enter for 'The cat runs'): ").strip()
    except EOFError:
        text = ""
    if not text:
        text = "The cat runs"
    return text


# step 5: tokenization
def tokenizeText(text):
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return tokens


# initial probabilities
def calcInitialProbabilities(corpus, tags):
    start_counts = Counter(sentence[0][1] for sentence in corpus)
    total_sentences = len(corpus)
    initial_probs = {tag: start_counts[tag] / total_sentences for tag in tags}
    return initial_probs


# transition probabilities
def calcTransitionProbabilities(corpus, tags):
    trans_counts = defaultdict(Counter)
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            t1 = sentence[i][1]
            t2 = sentence[i + 1][1]
            trans_counts[t1][t2] += 1

    trans_probs = defaultdict(dict)
    for t1 in tags:
        total_transitions = sum(trans_counts[t1].values())
        for t2 in tags:
            if total_transitions > 0:
                trans_probs[t1][t2] = trans_counts[t1][t2] / total_transitions
            else:
                trans_probs[t1][t2] = 0.0
    return trans_probs


# emission probabilities
def calcEmissionProbabilities(corpus, tags, vocab):
    emit_counts = defaultdict(Counter)
    tag_counts = Counter()
    for sentence in corpus:
        for word, tag in sentence:
            emit_counts[tag][word] += 1
            tag_counts[tag] += 1

    emit_probs = defaultdict(dict)
    for tag in tags:
        for word in vocab:
            if tag_counts[tag] > 0:
                emit_probs[tag][word] = emit_counts[tag][word] / tag_counts[tag]
            else:
                emit_probs[tag][word] = 0.0
    return emit_probs


# display tables
def displayProbabilities(initial_probs, trans_probs, emit_probs, tags, vocab):
    # initial table
    df_init = pd.DataFrame(
        list(initial_probs.items()), columns=["POS Tag", "Initial Probability"]
    )
    print("\n--- INITIAL PROBABILITIES ---")
    print(tabulate(df_init, headers="keys", tablefmt="psql", showindex=False))

    # transition table
    trans_matrix = []
    for t1 in tags:
        row = [t1] + [trans_probs[t1][t2] for t2 in tags]
        trans_matrix.append(row)
    df_trans = pd.DataFrame(trans_matrix, columns=["From / To"] + tags)
    print("\n--- TRANSITION PROBABILITIES ---")
    print(tabulate(df_trans, headers="keys", tablefmt="psql", showindex=False))

    # emission table
    emit_matrix = []
    for tag in tags:
        row = [tag] + [emit_probs[tag][word] for word in vocab]
        emit_matrix.append(row)
    df_emit = pd.DataFrame(emit_matrix, columns=["Tag / Word"] + vocab)
    print("\n--- EMISSION PROBABILITIES ---")
    print(tabulate(df_emit, headers="keys", tablefmt="psql", showindex=False))


# step 6: viterbi
def viterbi(tokens, tags, initial_probs, trans_probs, emit_probs):
    viterbi_table = []
    backpointer = []

    # forward step
    first_v = {}
    first_bp = {}
    for tag in tags:
        first_v[tag] = initial_probs[tag] * emit_probs[tag].get(tokens[0], 0.0)
        first_bp[tag] = None
    viterbi_table.append(first_v)
    backpointer.append(first_bp)

    for t in range(1, len(tokens)):
        current_v = {}
        current_bp = {}
        word = tokens[t]
        for tag in tags:
            best_prob, best_tag = max(
                (viterbi_table[t - 1][prev_tag] * trans_probs[prev_tag][tag], prev_tag)
                for prev_tag in tags
            )
            current_v[tag] = best_prob * emit_probs[tag].get(word, 0.0)
            current_bp[tag] = best_tag
        viterbi_table.append(current_v)
        backpointer.append(current_bp)

    # backtrack path
    best_last_tag = max(tags, key=lambda tag: viterbi_table[-1][tag])
    best_path = [best_last_tag]
    for t in range(len(tokens) - 1, 0, -1):
        best_last_tag = backpointer[t][best_last_tag]
        best_path.append(best_last_tag)
    best_path.reverse()

    return best_path, viterbi_table


# step 7: display
def displayViterbiResults(tokens, best_path, viterbi_table, tags):
    # trellis table
    trellis_data = []
    for t, word in enumerate(tokens):
        row = [word] + [viterbi_table[t][tag] for tag in tags]
        trellis_data.append(row)
    df_trellis = pd.DataFrame(trellis_data, columns=["Word"] + tags)
    print("\n--- VITERBI TRELLIS ---")
    print(tabulate(df_trellis, headers="keys", tablefmt="psql", showindex=False))

    # tagged table
    tagged_data = list(zip(tokens, best_path))
    df_tagged = pd.DataFrame(tagged_data, columns=["Word", "Predicted Tag"])
    print("\n--- PREDICTED TAGS ---")
    print(tabulate(df_tagged, headers="keys", tablefmt="psql", showindex=False))

    # formatted string
    formatted_seq = " ".join(f"{word}/{tag}" for word, tag in tagged_data)
    print("\n--- FINAL RESULT ---")
    print(formatted_seq)


# main function
def main():
    print("Performing Installations...")
    installations()
    time.sleep(1)
    print("Installation Successful!\n")

    # get corpus
    corpus = getCorpus()
    tags = sorted(list({tag for sent in corpus for _, tag in sent}))
    vocab = sorted(list({word for sent in corpus for word, _ in sent}))

    # calculate probabilities
    initial_probs = calcInitialProbabilities(corpus, tags)
    trans_probs = calcTransitionProbabilities(corpus, tags)
    emit_probs = calcEmissionProbabilities(corpus, tags, vocab)

    # display probabilities
    time.sleep(1)
    displayProbabilities(initial_probs, trans_probs, emit_probs, tags, vocab)

    # get input
    time.sleep(1)
    print("\n--- INPUT SENTENCE ---")
    text = getInput()

    # tokenize input
    tokens = tokenizeText(text)
    time.sleep(1)
    print("\n--- WORD TOKENS ---")
    print(tokens)

    # run viterbi
    time.sleep(1)
    best_path, viterbi_table = viterbi(
        tokens, tags, initial_probs, trans_probs, emit_probs
    )

    # show results
    time.sleep(1)
    displayViterbiResults(tokens, best_path, viterbi_table, tags)


# run program
if __name__ == "__main__":
    main()
