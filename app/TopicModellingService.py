import gensim
import gensim.corpora as corpora
import nltk
import spacy
from nltk.corpus import stopwords
import pandas as pd
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")


def topic_modelling(decription_text, model):
    event_desc = preprocess(decription_text)
    bigram = gensim.models.Phrases(event_desc, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    processed_docs_bigrams = bigram_mod[event_desc]
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(processed_docs_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    df_topic_sents_keywords = format_topics_sentences(ldamodel=model, corpus=corpus,
                                                           texts=data_lemmatized)
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return df_dominant_topic.Dominant_Topic.values


# Function to lemmatize the given tokens
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    doc = nlp(" ".join(texts))
    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Function to tokenize the given text
def preprocess(text):
    # Get NLTK stopwords for English and Dutch
    stopwords_dutch = stopwords.words('dutch')
    stopwords_all = stopwords.words('english')

    # Extend the stopword list with the common occurences of the topic
    stopwords_all.extend(['join', 'meet', 'event', 'attend', 'time', 'day', 'week', 'group', 'pm', 'am',
                          'rsvp', 'come', 'register', 'contact', 'welcome', 'member', 'session',
                          'schedule', 'get', 'meetup', 'th', 'yet', 'also', 'de', 'let', 'lets', 'events', 'able',
                          'via'])

    # Extend the list with Dutch stopwords
    stopwords_all.extend(stopwords_dutch)
    result = []
    for token in gensim.utils.simple_preprocess(text, deacc=True):
        if token not in stopwords_all and len(token) > 3:
            result.append(token)
    return result


# function to find the topic number that has the highest percentage contribution in that document.
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:

                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df
