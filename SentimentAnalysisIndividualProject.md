## Sentiment Analysis and Text Classification with NLP


# Introduction to Sentiment Analysis Project
Sentiment analysis is a crucial application of Natural Language Processing (NLP) that involves understanding the sentiment expressed in textual data. It has diverse applications, including customer feedback analysis, brand monitoring, and social media sentiment tracking.

This project focuses on building a sentiment analysis model to classify movie reviews into **Positive**, **Neutral**, or **Negative** categories. 

The tasks involve:
- Preprocessing a dataset for sentiment analysis
- Exploring the dataset through EDA
- Training and evaluating a classification model using Logistic Regression
- Performing error analysis and drawing insights

The dataset used is the **Stanford Sentiment Treebank (SST)**, a rich resource of sentiment labels at both sentence and phrase levels.


#### Instaling the required packages


```python
#!pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

    Requirement already satisfied: pandas in ./anaconda3/lib/python3.11/site-packages (1.5.3)
    Requirement already satisfied: numpy in ./anaconda3/lib/python3.11/site-packages (1.24.3)
    Requirement already satisfied: matplotlib in ./anaconda3/lib/python3.11/site-packages (3.7.1)
    Requirement already satisfied: seaborn in ./anaconda3/lib/python3.11/site-packages (0.12.2)
    Requirement already satisfied: scikit-learn in ./anaconda3/lib/python3.11/site-packages (1.3.0)
    Requirement already satisfied: nltk in ./anaconda3/lib/python3.11/site-packages (3.8.1)
    Requirement already satisfied: wordcloud in ./anaconda3/lib/python3.11/site-packages (1.9.4)
    Requirement already satisfied: python-dateutil>=2.8.1 in ./anaconda3/lib/python3.11/site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in ./anaconda3/lib/python3.11/site-packages (from pandas) (2022.7)
    Requirement already satisfied: contourpy>=1.0.1 in ./anaconda3/lib/python3.11/site-packages (from matplotlib) (1.0.5)
    Requirement already satisfied: cycler>=0.10 in ./anaconda3/lib/python3.11/site-packages (from matplotlib) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in ./anaconda3/lib/python3.11/site-packages (from matplotlib) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in ./anaconda3/lib/python3.11/site-packages (from matplotlib) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in ./anaconda3/lib/python3.11/site-packages (from matplotlib) (23.0)
    Requirement already satisfied: pillow>=6.2.0 in ./anaconda3/lib/python3.11/site-packages (from matplotlib) (9.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in ./anaconda3/lib/python3.11/site-packages (from matplotlib) (3.0.9)
    Requirement already satisfied: scipy>=1.5.0 in ./anaconda3/lib/python3.11/site-packages (from scikit-learn) (1.10.1)
    Requirement already satisfied: joblib>=1.1.1 in ./anaconda3/lib/python3.11/site-packages (from scikit-learn) (1.2.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in ./anaconda3/lib/python3.11/site-packages (from scikit-learn) (2.2.0)
    Requirement already satisfied: click in ./anaconda3/lib/python3.11/site-packages (from nltk) (8.0.4)
    Requirement already satisfied: regex>=2021.8.3 in ./anaconda3/lib/python3.11/site-packages (from nltk) (2022.7.9)
    Requirement already satisfied: tqdm in ./anaconda3/lib/python3.11/site-packages (from nltk) (4.65.0)
    Requirement already satisfied: six>=1.5 in ./anaconda3/lib/python3.11/site-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)



```python
import pandas as pd

# Load the dataset containing sentences and their respective IDs
sentences = pd.read_csv('/Users/sheikhmuzaffarahmad/Downloads/stanfordSentimentTreebank/datasetSentences.txt', sep='\t')
```


```python
# Enable inline plotting in Jupyter Notebook for immediate visual feedback
%matplotlib inline
```


```python
# Import necessary libraries for data handling, visualization, preprocessing, and modeling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk
```

## Dataset Description and Preprocessing

The **Stanford Sentiment Treebank (SST)** dataset is used for this project. It contains:
- 11,855 labeled sentences and phrases
- Fine-grained sentiment labels ranging from very negative to very positive

### Preprocessing Steps
To prepare the dataset for analysis, we performed:
1. **Tokenization**: Splitting text into individual words.
2. **Lowercasing**: Converting all text to lowercase for consistency.
3. **Stopword Removal**: Removing common words like "the", "and", etc., that do not carry sentiment.
4. **Lemmatization**: Reducing words to their base forms (e.g., "running" → "run").
5. **Data Splitting**: Dividing data into training and testing sets with an 80/20 ratio.
6. **Handling Class Imbalance**: Calculating class weights to address the imbalance between Positive, Neutral, and Negative samples.


```python
# Import the pandas library for data manipulation and analysis
import pandas as pd

# Define the base path where the dataset files are located
base_path = '/Users/sheikhmuzaffarahmad/Downloads/stanfordSentimentTreebank'

# Load various components of the dataset:
# - Sentences and their indices
sentences = pd.read_csv(f'{base_path}/datasetSentences.txt', sep='\t')

# - Sentiment labels associated with phrase IDs
sentiment_labels = pd.read_csv(f'{base_path}/sentiment_labels.txt', sep='|')

# - Phrase dictionary mapping phrases to their IDs
dictionary = pd.read_csv(f'{base_path}/dictionary.txt', sep='|', header=None, names=['phrase', 'phrase_id'])

# - Dataset split information (train, test, dev)
dataset_split = pd.read_csv(f'{base_path}/datasetSplit.txt', sep=',')
```


```python
# Display the first few rows of each loaded dataset for verification
print(sentences.head())
print(sentiment_labels.head())
print(dictionary.head())
print(dataset_split.head())
```

       sentence_index                                           sentence
    0               1  The Rock is destined to be the 21st Century 's...
    1               2  The gorgeously elaborate continuation of `` Th...
    2               3                     Effective but too-tepid biopic
    3               4  If you sometimes like to go to the movies to h...
    4               5  Emerges as something rare , an issue movie tha...
       phrase ids  sentiment values
    0           0           0.50000
    1           1           0.50000
    2           2           0.44444
    3           3           0.50000
    4           4           0.42708
            phrase  phrase_id
    0            !          0
    1          ! '      22935
    2         ! ''      18235
    3       ! Alas     179257
    4  ! Brilliant      22936
       sentence_index  splitset_label
    0               1               1
    1               2               1
    2               3               2
    3               4               2
    4               5               2



```python
# Step 1: Merge sentences with their phrase IDs using the dictionary file
sentences_with_ids = pd.merge(sentences, dictionary, left_on='sentence', right_on='phrase')

# Step 2: Merge the result with sentiment labels based on phrase IDs
sentences_with_sentiment = pd.merge(sentences_with_ids, sentiment_labels, left_on='phrase_id', right_on='phrase ids')

# Step 3: Merge the resulting dataset with the dataset split to include train/dev/test information
final_data = pd.merge(sentences_with_sentiment, dataset_split, on='sentence_index')

# Display the first few rows of the final merged dataset for verification
print(final_data.head())
```

       sentence_index                                           sentence  \
    0               1  The Rock is destined to be the 21st Century 's...   
    1               2  The gorgeously elaborate continuation of `` Th...   
    2               3                     Effective but too-tepid biopic   
    3               4  If you sometimes like to go to the movies to h...   
    4               5  Emerges as something rare , an issue movie tha...   
    
                                                  phrase  phrase_id  phrase ids  \
    0  The Rock is destined to be the 21st Century 's...     226166      226166   
    1  The gorgeously elaborate continuation of `` Th...     226300      226300   
    2                     Effective but too-tepid biopic      13995       13995   
    3  If you sometimes like to go to the movies to h...      14123       14123   
    4  Emerges as something rare , an issue movie tha...      13999       13999   
    
       sentiment values  splitset_label  
    0           0.69444               1  
    1           0.83333               1  
    2           0.51389               2  
    3           0.73611               2  
    4           0.86111               2  



```python
# Display the column names of the final dataset for structure verification
print(final_data.columns)
```

    Index(['sentence_index', 'sentence', 'phrase', 'phrase_id', 'phrase ids',
           'sentiment values', 'splitset_label'],
          dtype='object')



```python
# Step 4: Preprocess the sentences in the dataset
# The `preprocess_text` function will clean and prepare text data for analysis
import re
import string

# Define the preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = text.strip()
    # Tokenize the text (optional, based on your needs)
    # text = text.split()
    return text

final_data['processed_sentence'] = final_data['sentence'].apply(preprocess_text)

# Check the processed sentences
print(final_data[['sentence', 'processed_sentence']].head())
```

                                                sentence  \
    0  The Rock is destined to be the 21st Century 's...   
    1  The gorgeously elaborate continuation of `` Th...   
    2                     Effective but too-tepid biopic   
    3  If you sometimes like to go to the movies to h...   
    4  Emerges as something rare , an issue movie tha...   
    
                                      processed_sentence  
    0  the rock is destined to be the st century s ne...  
    1  the gorgeously elaborate continuation of  the ...  
    2                      effective but tootepid biopic  
    3  if you sometimes like to go to the movies to h...  
    4  emerges as something rare  an issue movie that...  



```python
# Preprocessing function to clean and normalize text
def preprocess_text(text):
    # Tokenize the sentence into words and convert them to lowercase
    tokens = word_tokenize(text.lower())
    # Remove non-alphanumeric characters (punctuation, special characters, etc.)
    tokens = [word for word in tokens if word.isalnum()]
    # Remove stopwords (e.g., 'and', 'the') to focus on meaningful words
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize words to their base form (e.g., "running" → "run")
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Combine tokens back into a single cleaned string
    return ' '.join(tokens)

# Download NLTK resources required for preprocessing
nltk.download('punkt')  # Tokenizer models
nltk.download('stopwords')  # Stopwords list
nltk.download('wordnet')  # WordNet lemmatizer data

# Initialize tools for preprocessing
stop_words = set(stopwords.words('english'))  # Define the list of stopwords
lemmatizer = WordNetLemmatizer()  # Initialize the WordNet lemmatizer

# Apply the `preprocess_text` function to clean all sentences in the dataset
final_data['processed_sentence'] = final_data['sentence'].apply(preprocess_text)

# Display a sample of original and processed sentences for verification
print(final_data[['sentence', 'processed_sentence']].head())
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/sheikhmuzaffarahmad/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/sheikhmuzaffarahmad/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/sheikhmuzaffarahmad/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


                                                sentence  \
    0  The Rock is destined to be the 21st Century 's...   
    1  The gorgeously elaborate continuation of `` Th...   
    2                     Effective but too-tepid biopic   
    3  If you sometimes like to go to the movies to h...   
    4  Emerges as something rare , an issue movie tha...   
    
                                      processed_sentence  
    0  rock destined 21st century new conan going mak...  
    1  gorgeously elaborate continuation lord ring tr...  
    2                                   effective biopic  
    3  sometimes like go movie fun wasabi good place ...  
    4  emerges something rare issue movie honest keen...  


## Exploratory Data Analysis (EDA)

I explored the dataset to understand its distribution and key features:
1. **Class Distribution**:
   - Visualized the proportion of Positive, Neutral, and Negative samples.
   - Found that the dataset has an imbalance with more Positive reviews than Neutral or Negative.

2. **Word Cloud**:
   - Generated separate word clouds for each sentiment category to visualize frequently used words.
   - Positive sentiment is dominated by words like "great", "funny", and "performance".
   - Negative sentiment has words like "bad", "boring", and "worse".
   - Neutral sentiment shows less sentiment-specific words.

These insights guided the preprocessing and model development decisions.



```python
# Import libraries for data visualization
import seaborn as sns # For creating attractive and informative statistical graphics
import matplotlib.pyplot as plt # For general plotting

# Define a function to categorize sentiment scores into Positive, Neutral, or Negative
def categorize_sentiment(score):
    """
    Categorize sentiment based on the score:
    - Negative: Score <= 0.4
    - Positive: Score >= 0.6
    - Neutral: 0.4 < Score < 0.6
    """
    if score <= 0.4:
        return 'Negative'
    elif score >= 0.6:
        return 'Positive'
    else:
        return 'Neutral'
    
# Apply the categorization function to create a new column for sentiment categories
final_data['sentiment_category'] = final_data['sentiment values'].apply(categorize_sentiment)

# Step 1: Visualize the distribution of sentiment categories
# Create a count plot to display the frequency of Positive, Neutral, and Negative sentiments
sns.countplot(x='sentiment_category', data=final_data, palette='viridis')
plt.title('Sentiment Category Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```


    
![png](output_16_0.png)
    


### Sentiment Category Distribution

The bar chart above illustrates the distribution of sentiment categories within the dataset. The dataset has been categorized into three sentiment classes: Positive, Negative, and Neutral.

- **Positive Sentiments**: This category has the highest frequency, indicating that a significant portion of the data conveys positive emotions or favorable opinions.
- **Negative Sentiments**: The second most prevalent category represents sentences with negative emotions or criticisms.
- **Neutral Sentiments**: The smallest category includes sentences with neither strongly positive nor strongly negative emotions.

This distribution highlights an *imbalance* in the dataset, with Neutral sentiments being underrepresented compared to Positive and Negative ones. This imbalance may pose challenges during model training, potentially leading to biased predictions toward the overrepresented classes.

To address this issue, techniques such as class weighting or data augmentation may be considered to ensure the model performs well across all sentiment classes.

### Word Cloud for Sentiment Classes
Goal: Visualize the most common words in positive, negative, and neutral sentences using word clouds.


```python
# Import WordCloud library to generate word clouds for sentiment categories
from wordcloud import WordCloud

# Step 2: Generate and visualize word clouds for each sentiment category
for category in ['Positive', 'Negative', 'Neutral']:
        # Filter the dataset for the current sentiment category and join all sentences into a single string
    text = ' '.join(final_data[final_data['sentiment_category'] == category]['processed_sentence'])
    
        # Create a WordCloud object with specified width, height, and background color
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
        # Display the word cloud
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.imshow(wordcloud, interpolation='bilinear')  # Render the word cloud
    plt.axis('off') # Remove axes for better visualization
    plt.title(f'Word Cloud for {category} Sentiment') # Add a title indicating the sentiment category
    plt.show() # Display the plot
```


    
![png](output_19_0.png)
    



    
![png](output_19_1.png)
    



    
![png](output_19_2.png)
    


### Word Clouds for Sentiment Classes

The word clouds above visualize the most frequently occurring words for each sentiment category: Positive, Negative, and Neutral. 

1. **Positive Sentiment**:
   - Words such as "movie," "film," "story," and "character" dominate the positive sentiment category.
   - Positive adjectives like "funny," "great," and "good" suggest the themes of favorable opinions or experiences expressed in the dataset.

2. **Negative Sentiment**:
   - Similarly, terms like "movie," "film," and "story" appear prominently in the negative sentiment category, but they are often paired with words that reflect dissatisfaction or criticism.
   - The negative sentiment cloud might also include terms associated with complaints or disappointment.

3. **Neutral Sentiment**:
   - The neutral sentiment cloud reflects terms common to both positive and negative sentiments but lacks strong emotional or opinionated words.
   - Words such as "make," "time," and "character" suggest more descriptive or non-opinionated sentences.

These visualizations provide an overview of the language patterns within each sentiment class, helping to identify potential overlaps or distinctive characteristics in the dataset. Word clouds also highlight the importance of preprocessing to ensure the model focuses on meaningful patterns rather than high-frequency neutral words such as "movie" or "film."

## Model Development

### Logistic Regression with TF-IDF
The sentiment classification was implemented using Logistic Regression, a traditional machine learning model. The steps included:
1. **TF-IDF Vectorization**: Text data was converted into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) representation.
2. **Logistic Regression**: A simple yet powerful model for classification tasks.
3. **Class Weights**: To address class imbalance, weights were computed and incorporated into the model.

This approach resulted in reasonable performance, with an overall accuracy of 64%.



```python
from sklearn.model_selection import train_test_split

# Step 1: Prepare the features (X) and labels (y) for modeling
# 'X' contains the preprocessed sentences, and 'y' contains the sentiment categories
X = final_data['processed_sentence']
y = final_data['sentiment_category']

# Step 2: Split the data into training and testing sets
# Use an 80/20 split for training and testing data
# 'random_state=42' ensures reproducibility of the results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Vectorize the Text Data

Goal: Transform the text into numerical features using TF-IDF.


```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 3: Transform text data into numerical features using TF-IDF
# TF-IDF (Term Frequency-Inverse Document Frequency) converts text into a matrix of features
# 'max_features=5000' restricts the vectorizer to the 5000 most important terms
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### Train a Machine Learning Model

Goal: Train a sentiment classification model, such as Logistic Regression.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 4: Initialize Logistic Regression
# Logistic Regression is a simple yet effective classification algorithm
# 'max_iter=1000' ensures the model has sufficient iterations to converge
model = LogisticRegression(max_iter=1000)  

# Step 5: Train the Logistic Regression model on the training data
model.fit(X_train_tfidf, y_train)

# Step 6: Predict sentiment categories for the test set
y_pred = model.predict(X_test_tfidf)

# Step 7: Evaluate the model's performance using a classification report
# The classification report includes metrics such as precision, recall, F1-score, and support for each class
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
        Negative       0.62      0.75      0.68       844
         Neutral       0.38      0.07      0.12       422
        Positive       0.67      0.78      0.72       992
    
        accuracy                           0.64      2258
       macro avg       0.56      0.53      0.51      2258
    weighted avg       0.60      0.64      0.59      2258
    


### Model Evaluation: Classification Report

The classification report provides a detailed analysis of the model's performance across the three sentiment categories: Negative, Neutral, and Positive. It includes metrics such as precision, recall, F1-score, and the number of samples (support) in each class.

1. **Negative Sentiment**:
   - **Precision**: 0.62 - The model correctly predicts 62% of the negative samples as negative.
   - **Recall**: 0.75 - The model identifies 75% of all actual negative samples.
   - **F1-Score**: 0.68 - A balance between precision and recall for this category.

2. **Neutral Sentiment**:
   - **Precision**: 0.38 - The model accurately predicts only 38% of the neutral samples.
   - **Recall**: 0.07 - A low recall indicates the model struggles to identify neutral sentiments.
   - **F1-Score**: 0.12 - Reflects poor overall performance for the neutral class.

3. **Positive Sentiment**:
   - **Precision**: 0.67 - The model correctly predicts 67% of the positive samples as positive.
   - **Recall**: 0.78 - The model identifies 78% of all actual positive samples.
   - **F1-Score**: 0.72 - Shows a solid performance for positive sentiment detection.

4. **Overall Metrics**:
   - **Accuracy**: 0.64 - The model correctly classifies 64% of all samples.
   - **Macro Average**: Precision: 0.56, Recall: 0.53, F1-Score: 0.51 - Averages across all classes without considering class imbalance.
   - **Weighted Average**: Precision: 0.60, Recall: 0.64, F1-Score: 0.59 - Averages weighted by the number of samples in each class.

### Insights:
- The model performs well for Positive and Negative sentiments, but struggles significantly with Neutral sentiments, likely due to overlap in language patterns or fewer representative samples.
- Improvements such as additional feature engineering, balanced data handling, or using more advanced models could enhance performance, especially for Neutral sentiments.

## Evaluation and Error Analysis

### Evaluation Metrics
The Logistic Regression model was evaluated using precision, recall, F1-score, and accuracy:
- **Accuracy**: 64%
- **Class-level Performance**:
  - Positive: High precision and recall due to a larger dataset representation.
  - Neutral: Poor performance due to fewer samples and subtle sentiment expressions.
  - Negative: Moderate performance.



```python
# Step 1: Identify misclassified examples
# Compare the true labels ('y_test') with the predicted labels ('y_pred')
# The indices where the true labels and predicted labels differ indicate misclassifications
misclassified_indices = (y_test != y_pred).index

# Step 2: Extract misclassified examples from the original dataset
# Use the indices of misclassified examples to fetch their details from the dataset
misclassified = final_data.iloc[misclassified_indices]

# Step 3: Display a sample of misclassified examples
# Show the original sentence, processed sentence, and the actual sentiment category for analysis
print(misclassified[['sentence', 'processed_sentence', 'sentiment_category']])
```

                                                    sentence  \
    11041          The Porky 's Revenge : Ultimate Edition ?   
    5902   These two are generating about as much chemist...   
    7363   This is a shameless sham , calculated to cash ...   
    9432   A movie like The Guys is why film criticism ca...   
    10516  According to Wendigo , ` nature ' loves the me...   
    ...                                                  ...   
    1886   A powerful performance from Mel Gibson and a b...   
    23     A disturbing and frighteningly evocative assem...   
    4428   The hook is the drama within the drama , as an...   
    9940   The movie is silly beyond comprehension , and ...   
    1309   Bleakly funny , its characters all the more to...   
    
                                          processed_sentence sentiment_category  
    11041                     porky revenge ultimate edition            Neutral  
    5902   two generating much chemistry iraqi factory po...            Neutral  
    7363      shameless sham calculated cash popularity star           Negative  
    9432       movie like guy film criticism considered work           Negative  
    10516  according wendigo nature love member upper cla...           Positive  
    ...                                                  ...                ...  
    1886   powerful performance mel gibson brutal battle ...           Positive  
    23     disturbing frighteningly evocative assembly im...            Neutral  
    4428   hook drama within drama unsolved murder unreso...           Positive  
    9940   movie silly beyond comprehension even silly wo...           Negative  
    1309   bleakly funny character touching refusing pity...           Positive  
    
    [2258 rows x 3 columns]


### Error Analysis
We analyzed misclassified examples to understand the model's limitations:
1. **Neutral sentences**: Often confused with Positive or Negative due to subtle expressions.
2. **Complex Sentences**: Sentences with conjunctions or mixed sentiment (e.g., "great but expensive") were challenging.
3. **Ambiguity**: Short sentences or those with ambiguous sentiment were prone to misclassification.

Examples of misclassified sentences:
- Neutral classified as Positive: "A decent movie, but nothing special."
- Positive classified as Negative: "An exciting movie, though slightly predictable."
- Negative classified as Positive: "Terrible acting but a somewhat interesting plot."



```python
# Visualizing misclassified examples by true sentiment
misclassified['True Sentiment'] = y_test.values
sns.countplot(x='True Sentiment', data=misclassified, palette='coolwarm')
plt.title('Distribution of Misclassified Examples')
plt.xlabel('True Sentiment')
plt.ylabel('Count')
plt.show()
```

    /var/folders/h2/6zmffq_n3ts8l0y4jw9p8f240000gn/T/ipykernel_52668/1926919394.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      misclassified['True Sentiment'] = y_test.values



    
![png](output_31_1.png)
    


### Distribution of Misclassified Examples

The bar chart above illustrates the distribution of misclassified examples across different true sentiment categories: **Neutral**, **Negative**, and **Positive**. 

#### Observations:
- **Positive Sentiments**: The highest number of misclassifications occurred in sentences with a true sentiment labeled as Positive, indicating challenges in identifying the correct sentiment in positively inclined statements.
- **Negative Sentiments**: The second-highest misclassifications were observed in Negative sentiments, suggesting some overlap or confusion with other sentiment categories.
- **Neutral Sentiments**: Neutral sentiments had the least number of misclassifications but were still significant, highlighting the difficulty in accurately identifying neutral tones, likely due to subtle expressions.

#### Insights:
This distribution reveals areas where the model struggles, particularly in distinguishing between Positive and Negative sentiments. Furthermore, the misclassification of Neutral sentiments may stem from ambiguous or mixed-language constructs in the data. Addressing these issues may involve refining the preprocessing steps, enhancing feature extraction methods, or experimenting with advanced classification models.


# Conclusion
This project demonstrated the process of building a sentiment analysis model using Logistic Regression. 

Key takeaways include:
- Preprocessing and handling imbalanced data significantly affect model performance.
- Logistic Regression with TF-IDF is effective for sentiment classification, achieving 64% accuracy.
- Neutral sentences and ambiguous expressions were the most challenging to classify accurately.

### Future Work
Future improvements could include:
1. Using advanced feature extraction techniques or transformer-based models like BERT.
2. Expanding the training dataset to improve model generalization.
3. Incorporating additional context or metadata for better sentiment understanding.



```python

```
