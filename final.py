# Import necessary libraries
import nltk  # Import the Natural Language Toolkit library for text processing
from nltk import word_tokenize, sent_tokenize  # Import functions for tokenizing text into words and sentences
from nltk.corpus import stopwords  # Import stopwords for filtering common words
from nltk.stem import WordNetLemmatizer  # Import a lemmatizer to reduce words to their base form
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Import the VADER sentiment analysis tool
from collections import Counter  # Import Counter for counting elements in lists
import re  # Import the regular expression library

# Function to read the 1st book
def read_book1():
    with open("2013.txt", "r", encoding="utf-8") as filename:
        my_file1 = filename.read()
        return my_file1  # Read and return the content of the first book

# Function to read the 2nd book
def read_book2():
    with open("2014.txt", "r", encoding="utf-8") as filename2:
        my_file2 = filename2.read()
        return my_file2  # Read and return the content of the second book

# Function to read the 3rd book
def read_book3():
    with open("2015.txt", "r", encoding="utf-8") as filename3:
        my_file3 = filename3.read()
        return my_file3  # Read and return the content of the third book

# Function to tokenize words
def word_fragmentation(text):
    words = nltk.word_tokenize(text)
    return words  # Tokenize the input text into individual words and return the list of words

# Function to remove stop words and normalize the text
def normalize_words(text):
    words = word_fragmentation(text)  # Tokenize the text into words
    stop_words = set(stopwords.words("english"))  # Get a set of English stopwords
    lemmatizer = WordNetLemmatizer()  # Initialize a lemmatizer
    real_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    # Normalize words by lemmatization and remove stopwords
    return real_words

# Function to remove punctuation and colons
def remove_punctuation(text):
    words = normalize_words(text)  # Tokenize and normalize the text
    punctuation_remove = [re.sub(r"[^\w\s]", "", word) for word in words]  # Remove punctuation from words
    punctuation_removed = [word.rstrip(":") for word in punctuation_remove]  # Remove trailing colons from words
    return punctuation_removed

# Function to perform sentiment analysis on text
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()  # Initialize the VADER sentiment analyzer
    chunk_size = 1000  # Define the size of text chunks for sentiment analysis
    sentiment_scores = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]  # Divide the text into chunks of the specified size
        sentiment_scores.append(analyzer.polarity_scores(chunk))  # Perform sentiment analysis on each chunk

    compound_scores = [score['compound'] for score in sentiment_scores]  # Extract the compound sentiment scores
    average_sentiment = sum(compound_scores) / len(compound_scores)  # Calculate the average sentiment score
    return average_sentiment  # Return the average sentiment score for the entire text

# Function to find keywords with extended context
def find_keywords_with_extended_context(text, keywords):
    keyword_context = {}
    sentences = sent_tokenize(text)  # Tokenize the text into sentences
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)  # Tokenize each sentence into words
        words = [word.lower() for word in words]  # Convert words to lowercase
        for keyword in keywords:
            keyword_indices = [i for i, word in enumerate(words) if word == keyword]  # Find indices of keyword occurrences
            for index in keyword_indices:
                start = max(0, index - 10)  # Define the starting index of the context
                end = min(len(words), index + 11)  # Define the ending index of the context
                context = ' '.join(words[start:end])  # Extract the context around the keyword
                keyword_context.setdefault(keyword, []).append(context)  # Store the keyword and its context
    return keyword_context  # Return a dictionary of keywords and their associated contexts

# Main function
def main():
    # Read the three books
    book1 = read_book1()  # Read the content of the first book
    book2 = read_book2()  # Read the content of the second book
    book3 = read_book3()  # Read the content of the third book

    # Perform sentiment analysis on each book
    sentiment_analysisbook1 = perform_sentiment_analysis(book1)  # Analyze sentiment of the first book
    sentiment_analysisbook2 = perform_sentiment_analysis(book2)  # Analyze sentiment of the second book
    sentiment_analysisbook3 = perform_sentiment_analysis(book3)  # Analyze sentiment of the third book

    # Print the sentiment analysis results for each book
    print("Sentiment Analysis for the Book 'The Bamboo Stalk' ", sentiment_analysisbook1)
    print("Sentiment Analysis for the Book 'Frankenstein in Baghdad' ", sentiment_analysisbook2)
    print("Sentiment Analysis for the Book 'The Italian' ", sentiment_analysisbook3)

    print("/n" * 3)  # Print newlines for separation

    # Define the keywords to search for
    keywords = ["dictator", "dictatorship", "regime"]

    # Find keywords with extended context in each book
    keyword_context_book1 = find_keywords_with_extended_context(book1, keywords)  # Find contexts for keywords in book 1
    keyword_context_book2 = find_keywords_with_extended_context(book2, keywords)  # Find contexts for keywords in book 2
    keyword_context_book3 = find_keywords_with_extended_context(book3, keywords)  # Find contexts for keywords in book 3

    # Print or analyze the context of each keyword occurrence
    for keyword, contexts in keyword_context_book1.items():
        print(f"Keyword book 1: {keyword}")
        for context in contexts:
            print(f"Context: {context}")

    print("/n" * 3)  # Print newlines for separation

    for keyword, contexts in keyword_context_book2.items():
        print(f"Keyword book 2: {keyword}")
        for context in contexts:
            print(f"Context: {context}")

    print("/n" * 3)  # Print newlines for separation

    for keyword, contexts in keyword_context_book3.items():
        print(f"Keyword book 3: {keyword}")
        for context in contexts:
            print(f"Context: {context}")

if __name__ == "__main__":
    main()  # Execute the main function when the script is run

