import pandas as pd
import nltk
import string
import os
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tabulate import tabulate
from transformers import pipeline
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Settings
languages_map = {
    "en": "English",
    "de": "German",
    "ar": "Arabic"
}

# Load data
df = pd.read_csv("judge_pairwise_evaluation.csv")

# Justification columns
justification_cols = [
    'Justification gpt-4o-mini',
    'Justification gemini-2.5-flash',
    'Justification qwen-plus',
    'Justification claude-3-5-haiku'
]

def get_language_index(row, lang_code):
    for i in range(1, 4):
        if row[f'Solution {i} Language'].lower() == lang_code:
            return i
    return None

# Clean text
def clean(text, stop_words):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join([w for w in text.split() if w not in stop_words])

def get_top_ngrams(corpus, ngram_range=(1, 1), n=20):
    vec = CountVectorizer(ngram_range=ngram_range)
    X = vec.fit_transform(corpus)
    freqs = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    return sorted(zip(vocab, freqs), key=lambda x: x[1], reverse=True)[:n]


def print_ngrams(title, data):
    print(f"\n{title}")
    print(tabulate(data, headers=["N-gram", "Frequency"], tablefmt="github"))
    
def analyze_justifications_per_language(file_dir="", target_language = 'en'):
    df = pd.read_csv(os.path.join(file_dir, "judge_pairwise_evaluation.csv"))

    # Stopwords
    stop_words = list(stopwords.words("english"))
    custom_stopwords = [
        "solution", "en", "de", "ar", "ranks", "first", "second", "third",
        "ranked", "language", "english", "german", "arabic"
    ]
    stop_words.extend(custom_stopwords)

    df["Lang_Index"] = df.apply(lambda row: get_language_index(row, target_language), axis=1)

    # Extract sentences about the target language
    tokenizer = PunktSentenceTokenizer()
    target_sentences = []

    for _, row in df.iterrows():
        lang_idx = row["Lang_Index"]
        if not lang_idx:
            continue

        for col in justification_cols:
            text = row[col]
            if not isinstance(text, str):
                continue

            sentences = tokenizer.tokenize(text)
            for s in sentences:
                s_clean = s.lower()
                if f"The {languages_map[target_language]} solution" in s_clean or f"solution {lang_idx} (" in s_clean:
                    target_sentences.append(s.strip())


    cleaned_sentences = [clean(s, stop_words) for s in target_sentences if isinstance(s, str)]

    top_unigrams = get_top_ngrams(cleaned_sentences, (1, 1), 20)
    top_bigrams = get_top_ngrams(cleaned_sentences, (2, 2), 20)
    top_trigrams = get_top_ngrams(cleaned_sentences, (3, 3), 20)

    print_ngrams(f"Top Unigrams in {target_language.upper()} Justifications", top_unigrams)
    print_ngrams("Top Bigrams", top_bigrams)
    print_ngrams("Top Trigrams", top_trigrams)

    # Sentiment analysis using transformer
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    sentiment_results = sentiment_model(target_sentences)

    df_sentiment = pd.DataFrame(sentiment_results)
    df_sentiment["text"] = target_sentences
    df_sentiment["label"] = df_sentiment["label"].map({
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    })

    # Display and plot sentiment
    pd.set_option('display.max_colwidth', None)
    print("\nSample Sentiment-Labeled Sentences")
    print(df_sentiment[["text", "label"]].head())
    pd.reset_option('display.max_colwidth')

    print("\nSentiment Distribution:")
    print(df_sentiment["label"].value_counts())

    sns.countplot(data=df_sentiment, x="label", order=["negative", "neutral", "positive"])
    plt.title(f"Sentiment of Justifications About {target_language.upper()}")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Sentences")
    plt.tight_layout()
    plt.show()

    # Topic modeling
    vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(cleaned_sentences)
    lda = LatentDirichletAllocation(n_components=4, random_state=42)
    lda.fit(X)

    terms = vectorizer.get_feature_names_out()
    topics = []
    print("\nTop words per topic:")
    for idx, topic in enumerate(lda.components_):
        top_terms = [terms[i] for i in topic.argsort()[-8:][::-1]]
        topic_text = f"Topic {idx+1}: {', '.join(top_terms)}"
        topics.append(topic_text)
        print(topic_text)

    # Generate GPT-4o-mini-based summary
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    sentiment_summary = df_sentiment["label"].value_counts().to_dict()

    prompt = f"""
    You are an educational research assistant. Based on the following sentiment analysis and topic modeling of justifications provided by LLMs evaluating {target_language.upper()} math solutions, write a brief, analytical conclusion about the weaknesses of these solutions.

    Sentiment Breakdown: {sentiment_summary}

    Top Unigrams in Justifications:
    {top_unigrams}

    Top Bigrams in Justifications:
    {top_bigrams}

    Top Trigrams in Justifications:
    {top_trigrams}

    Topic Modeling:
    {chr(10).join(topics)}

    Write 3–5 sentences summarizing the strengths or the key shortcomings in {target_language.upper()} solutions, based on the evidence above.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nGPT-4o-mini Conclusion:")
    print(response.choices[0].message.content.strip())


if __name__ == "__main__":
    analyze_justifications_per_language(target_language='en')
