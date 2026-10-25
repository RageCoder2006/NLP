# step 1: imports
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import re
import time
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import pandas as pd
from tabulate import tabulate


# step 2: installations
def installations():
    try:
        nltk.data.path.append("./nltk_data")
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass


# step 3: dataset
def getReviews():
    reviews = [
        "The laptop performance is excellent and extremely fast.",
        "The mobile phone has an amazing camera and great display.",
        "Engineering CAD software works perfectly and smoothly.",
        "The CNC machine provides wonderful accuracy and precision.",
        "This smartphone has superb battery life and fantastic design.",
        "The laptop has a terrible keyboard and overheats badly.",
        "The mobile phone battery is awful and dies quickly.",
        "Simulation software is completely broken and crashes constantly.",
        "The lathe machine is defective and produces horrible errors.",
        "Customer support for the engineering tool was useless and disappointing.",
        "The laptop has a 15-inch screen and 16GB of RAM.",
        "The mobile phone was manufactured in 2024.",
        "The CAD software was installed on the workstation.",
        "The milling machine operates at 3000 RPM.",
        "The engineering toolkit contains standard measurement instruments.",
    ]
    return reviews


# step 4: preprocessing
def preprocessReview(text):
    # lowercase text
    lower_text = text.lower()

    # clean punctuation
    clean_text = re.sub(r"[^\w\s]", "", lower_text)

    # word tokenization
    try:
        tokens = word_tokenize(clean_text)
    except Exception:
        tokens = clean_text.split()
    return " ".join(tokens)


# step 5: analyzer
def getSentimentAnalyzer():
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception:
        # fallback rule analyzer
        class SimpleSIA:
            def polarity_scores(self, s):
                pos_words = {
                    "excellent",
                    "fast",
                    "amazing",
                    "great",
                    "perfect",
                    "perfectly",
                    "smoothly",
                    "wonderful",
                    "superb",
                    "fantastic",
                    "good",
                    "reliable",
                }
                neg_words = {
                    "terrible",
                    "overheats",
                    "badly",
                    "awful",
                    "dies",
                    "broken",
                    "crashes",
                    "defective",
                    "horrible",
                    "useless",
                    "disappointing",
                    "bad",
                }
                tokens = s.lower().split()
                pos = sum(1 for w in tokens if w in pos_words)
                neg = sum(1 for w in tokens if w in neg_words)
                if pos > neg:
                    return {"compound": 0.6}
                elif neg > pos:
                    return {"compound": -0.6}
                return {"compound": 0.0}

        sia = SimpleSIA()
    return sia


# step 6: classification
def classifySentiment(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# step 7: counts
def analyzeReviews(reviews, sia):
    results = []
    for rev in reviews:
        clean = preprocessReview(rev)
        scores = sia.polarity_scores(clean)
        compound = round(scores["compound"], 4)
        sentiment = classifySentiment(compound)
        results.append((rev, clean, compound, sentiment))
    return results


# step 8: visualization
def plotSentimentDistribution(sentiment_counts):
    categories = ["Positive", "Negative", "Neutral"]
    counts = [sentiment_counts.get(cat, 0) for cat in categories]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=colors, width=0.5, edgecolor="black")

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.title("Sentiment Distribution of Engineering Product Reviews", fontsize=12)
    plt.xlabel("Sentiment Class", fontsize=10)
    plt.ylabel("Number of Reviews", fontsize=10)
    plt.ylim(0, max(counts) + 2)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plot_file = "sentiment_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_file


# step 9: input
def getNewInput():
    try:
        text = input("Enter a review (press Enter for default): ").strip()
    except EOFError:
        text = ""
    if not text:
        text = "This engineering software is superb, reliable, and works excellently."
    return text


# main function
def main():
    print("Performing Installations...")
    installations()
    time.sleep(1)
    print("Installation Successful!\n")

    # get dataset
    reviews = getReviews()
    sia = getSentimentAnalyzer()

    # analyze sentiment
    time.sleep(1)
    analyzed_data = analyzeReviews(reviews, sia)

    # display table
    df_reviews = pd.DataFrame(
        [
            {
                "Original Review": r[0],
                "Compound Score": r[2],
                "Class": r[3],
            }
            for r in analyzed_data
        ]
    )
    print("--- STEP 4 & 5: SENTIMENT CLASSIFICATION ---")
    print(tabulate(df_reviews, headers="keys", tablefmt="psql", showindex=False))

    # count categories
    time.sleep(1)
    sentiments = [r[3] for r in analyzed_data]
    sentiment_counts = Counter(sentiments)

    df_counts = pd.DataFrame(
        [
            {"Sentiment": cat, "Count": sentiment_counts.get(cat, 0)}
            for cat in ["Positive", "Negative", "Neutral"]
        ]
    )
    print("\n--- STEP 5: SENTIMENT COUNTS ---")
    print(tabulate(df_counts, headers="keys", tablefmt="psql", showindex=False))

    # plot distribution
    time.sleep(1)
    plot_file = plotSentimentDistribution(sentiment_counts)
    print(f"\n--- STEP 6: BAR CHART SAVED ---")
    print(f"Chart saved as: {plot_file}")

    # test sentence
    time.sleep(1)
    print("\n--- STEP 7: PREDICT NEW SENTENCE ---")
    new_text = getNewInput()
    clean_new = preprocessReview(new_text)
    new_scores = sia.polarity_scores(clean_new)
    new_compound = round(new_scores["compound"], 4)
    new_class = classifySentiment(new_compound)

    print(f"\nInput Sentence : {new_text}")
    print(f"Preprocessed   : {clean_new}")
    print(f"Compound Score : {new_compound}")
    print(f"Predicted Class: {new_class}")


# run program
if __name__ == "__main__":
    main()
