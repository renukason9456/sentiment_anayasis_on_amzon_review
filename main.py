from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import os


dataset_path = "amazon_review_polarity_csv"


files = os.listdir(dataset_path)
print("Files in the dataset folder:", files)


train_file = os.path.join(dataset_path, "test.csv")
data = pd.read_csv(train_file, header=None, names=["label", "title", "review"])


data = data.head(500)


print(data.head())


def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


data['Polarity'] = data['review'].apply(get_sentiment)


def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

data['Sentiment'] = data['Polarity'].apply(classify_sentiment)


sentiment_counts = data['Sentiment'].value_counts()


plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Texts")
plt.show()


data.to_csv("sentiment_results_500_rows.csv", index=False)
print("Results saved to sentiment_results_500_rows.csv")
