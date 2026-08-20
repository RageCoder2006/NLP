import nltk
import re
import time

# installations
def installations():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# get input and lowercase
def inpLower():
    x = str(input("Enter Input Text: "))
    y=x.lower()
    return x,y

# remove numbers
def remNums(str_inp):
    clean = re.sub(r'\d+', '', str_inp)
    return clean

# remove punctuations
def remPuncts(str_inp):
    clean = re.sub(r'[^\w\s]', '', str_inp)
    return clean

# remove whitespaces
def remWhite(str_inp):
    clean = str_inp.strip()
    return clean

# tokenise
def tokenise(str_inp):
    tokens = nltk.word_tokenize(str_inp)
    cleaned_tokens = []
    for token in tokens:
        cleaned_tokens.append(token)
    return cleaned_tokens

# remove stopwords
def remStopwords(inp_tokens):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    clean_inp_tokens = []
    for token in inp_tokens:
        if token not in stop_words:
            clean_inp_tokens.append(token)
    return clean_inp_tokens

def sentTokenise(str_inp):
    sentences = nltk.sent_tokenize(str_inp)
    return sentences

def textPreprocess():
    print("Performing Installations...")
    installations()
    time.sleep(1)
    print("Installation Successful!")
    time.sleep(1)
    inp, lower = inpLower()
    # print("Original String Input:", inp)
    time.sleep(1)
    print("Lower Case String Input...")
    no_nums = remNums(lower)
    time.sleep(1)
    print("Removing Numbers...")
    no_puncts = remPuncts(no_nums)
    time.sleep(1)
    print("Removing Punctuations...")
    no_whitespaces = remWhite(no_puncts)
    time.sleep(1)
    print("Removing Whitespaces...")
    tokens = tokenise(no_whitespaces)
    time.sleep(1)
    print("Generating Input Word Tokens...")
    no_stopwords = remStopwords(tokens)
    time.sleep(1)
    print("Removing Stopwords...")
    sent_tokens = sentTokenise(lower)
    time.sleep(1)
    print("Generating Sentence Tokens for ORIGINAL TEXT...")
    time.sleep(1)
    print("Preprocessing Successful")
    return inp, lower, no_nums, no_puncts, no_whitespaces, tokens, no_stopwords, sent_tokens

def analysis(inp, lower, no_nums, no_puncts, no_whitespaces, tokens, no_stopwords, sent_tokens):
    print("ANALYSIS:")
    print("1. Number of words before and after preprocessing:")
    print("   - BEFORE:", len(lower))
    print("   - AFTER:", len(no_stopwords))
    print("2. Effect of lowercase conversion and punctuation removal:")
    print("   - Original String Input:", inp)
    print("   - Lower Case String Input:", lower)
    print("   - Input after removing punctuations:", no_puncts)
    print("3. Number of word tokens generated:", len(tokens))
    print("4. Number of sentence tokens generated:", len(sent_tokens))

def main():
    analysis(*textPreprocess())

if __name__ == "__main__":
    main()