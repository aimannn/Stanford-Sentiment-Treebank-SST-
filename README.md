# Stanford-Sentiment-Treebank-SST-
This project focuses on building a sentiment analysis model to classify movie reviews into Positive, Neutral, or Negative categories. 

Brief Description of the Dataset and Preprocessing Steps

The Stanford Sentiment Treebank dataset comprises over 11,000 movie review sentences, parsed into binary tree structures. Each sentence is assigned a numerical sentiment score ranging from strongly negative to strongly positive, making it an ideal benchmark for sentiment analysis.

Key features of the dataset include:

• Sentences: Individual movie review sentences and their phrase IDs.
• Sentiment Scores: Numerical scores from 0 to 1, reflecting sentiment polarity.
• Dataset Splits: Predefined splits for training, testing, and validation.
• Phrase Dictionary: A mapping of phrases to unique IDs for merging labels.

Preprocessing Steps:

1. Merging Data: Combined sentences with sentiment scores using phrase IDs and dataset splits.
2. Categorization: Classified sentiments into Positive, Neutral, and Negative using score
thresholds.
3. Text Cleaning: Performed tokenization, lowercasing, removal of stop words and non-
alphanumeric characters, and lemmatization.
4. Vectorization: Applied TF-IDF to transform text into numerical features, ensuring clean and
structured data for model training.

Dataset Source: https://nlp.stanford.edu/sentiment/
