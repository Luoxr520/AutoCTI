
"""
特征生成模块
"""
import argparse

import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

pd.set_option("display.max_colwidth", 200)
import re
from tqdm import tqdm
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from textblob import TextBlob
from statistics import mean
import numpy as np
from readability import Readability

en_stops = set(stopwords.words('english'))

import spacy
nlp = spacy.load('en_core_web_lg')

import neuralcoref
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

model = None
model_domain = None
sbert_model = None
elmo = None
model_cyber = None

def main(params):
    print(params)

    input_name = params.dataDir + params.input
    output_name = params.dataDir + params.output

    print("### Start Prepare Dataset")
    print("Input: " + input_name)
    print("Output: " + output_name + "\n")

    df = pd.read_excel(input_name, engine="openpyxl", sheet_name="Sheet", header=0)
    input_rows = df.shape[0]
    print("Read input file. Rows: " + str(input_rows))


    # pre-process
    tqdm.pandas(desc="Pre-processing topic")
    df['topic_processed'] = df['topic'].progress_map(preprocess_data)

    tqdm.pandas(desc="Pre-processing content")
    df['content_processed'] = df['content'].progress_map(preprocess_data)

    # generate words count column
    nltk.download('punkt')
    df['topic_words_count'] = df['topic'].str.split().str.len()
    df['content_words_count'] = df['content'].str.split().str.len()
    df['total_words_count'] = df['topic_words_count'] + df['content_words_count']
    df['total_sentence_count'] = df.progress_apply(cal_sentence_count, axis=1)
    df['total_paragraph_count'] = df.progress_apply(cal_paragraph_count, axis=1)

    # generate mean columns
    df['mean_words_per_sentence'] = df['total_words_count'] / df['total_sentence_count']
    df['mean_words_per_paragraph'] = df['total_words_count'] / df['total_paragraph_count']
    df["mean_sent_per_paragraph"] = df['total_sentence_count'] / df['total_paragraph_count']

    # calculate standard deviation columns
    df["stdev_words_per_sentence"] = df.progress_apply(cal_stdev_per_sentence, axis=1)
    df["stdev_words_per_paragraph"] = df.progress_apply(cal_stdev_per_paragraph, axis=1)
    df["stdev_sent_per_paragraph"] = df.progress_apply(cal_stdev_sent_per_paragraph, axis=1)

    # Sentiment Score
    tqdm.pandas(desc="Generate sentiment for topic")
    df['sentiment_topic_pd'] = df['topic_processed'].progress_map(cal_sentiment)

    tqdm.pandas(desc="Generate sentiment for content")
    df['sentiment_content_pd'] = df['content_processed'].progress_map(cal_sentiment)

    # Jaccard Coefficient
    tqdm.pandas(desc="Generate Jaccard Coefficient")
    df['jaccard_coef_pd'] = df.progress_apply(cal_jaccard_coef, axis=1)

    # Cosine Similarity using scikit-learn on Raw Data
    tqdm.pandas(desc="Generate Cosine Similarity using sklearn")
    df['cosine_similarity_sklearn'] = df.progress_apply(cal_cosine_similarity, axis=1)

    # Cosine Similarity using scikit-learn on Processed Data
    tqdm.pandas(desc="Generate Cosine Similarity using sklearn on processed data")
    df['cosine_similarity_sklearn_pd'] = df.progress_apply(cal_cosine_similarity_on_processed_data, axis=1)

    # Cosine Similarity using spaCy on Raw Data
    tqdm.pandas(desc="Generate Cosine Similarity using spaCy")
    df['cosine_similarity_spacy'] = df.progress_apply(cal_cosine_similarity_spacy, axis=1)

    # Cosine Similarity using spaCy on Processed Data
    tqdm.pandas(desc="Generate Cosine Similarity using spaCy on processed data")
    df['cosine_similarity_spacy_pd'] = df.progress_apply(cal_cosine_similarity_spacy_on_processed_data, axis=1)

    # word mover's distance on Processed Data using GoogleNews-vectors-negative300
    print("Loading GoogleNews-vectors-negative300.bin.gz ...")
    global model
    model = KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin.gz', binary=True)

    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using GoogleNews-vectors-negative300")
    df['wmd_google_nonsplit_pd'] = df.progress_apply(cal_wmd, axis=1)

    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using GoogleNews-vectors-negative300")
    df['wmd_google_pd'] = df.progress_apply(cal_wmd_split, axis=1)

    # word mover's distance on Processed Data using Domain-word2vec
    print("Loading Domain-Word2vec.model.gz ...")
    global model_domain
    model_domain = Word2Vec.load('./model/Domain-Word2vec.model.gz')
    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using Domain-Word2vec")
    df['wmd_domain_pd'] = df.progress_apply(cal_wmd_domain, axis=1)

    # word mover's distance on Processed Data using Cyber-word2vec
    print("Loading Cyber-Word2vec/1million.word2vec.model ...")
    global model_cyber
    model_cyber = Word2Vec.load('./model/1million.word2vec.model')
    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using Cyber-Word2vec")
    df['wmd_cyber_nonsplit_pd'] = df.progress_apply(cal_wmd_cyber, axis=1)

    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using Cyber-Word2vec")
    df['wmd_cyber_pd'] = df.progress_apply(cal_wmd_cyber_split, axis=1)

    # readability with flesch reading ease
    tqdm.pandas(desc="Flesch Reading Ease")
    df['flesch_reading_ease'] = df.progress_apply(flesch_reading_ease, axis=1)

    # readability with flesch kincaid readability
    tqdm.pandas(desc="Flesch Kincaid Readability")
    df['flesch_kincaid_readability'] = df.progress_apply(flesch_kincaid, axis=1)

    # readability with gunning fog readability
    tqdm.pandas(desc="Gunning Fog Readability")
    df['gunning_fog_readability'] = df.progress_apply(gunning_fog, axis=1)

    # readability with automatic readability index
    tqdm.pandas(desc="Automatic Readability Index")
    df['automatic_readability_index'] = df.progress_apply(automatic_readability, axis=1)

    # Save
    df.to_excel(output_name, index=False)
    print("\n### Finish generate features.")
    print("Input: " + input_name)
    print("Output: " + output_name)
    print("Columns: " + str(df.columns.tolist()))

def remove_stopwords(Input):
    '''
    Removes stopwords from passed text
    '''
    output_list = []
    for word in Input.lower().split():
        if word not in en_stops:
            output_list.append(word)
    output = " ".join(output_list)
    return output

def replace_pronouns(text):
    '''
    Replaces pronouns in text
    '''
    doc = nlp(text)
    return doc._.coref_resolved

def preprocess_data(Input):
    '''
    Prepocesses text removing pronouns, stopwords and punctuation
    '''
    proc = Input.replace("\n", " ")
    proc = proc.replace("\r", "")

    # replacing pronouns
    proc = replace_pronouns(proc)

    # remove stopwords
    proc = remove_stopwords(proc)

    # remove punctuation
    output = proc.translate(str.maketrans('', '', string.punctuation))

    return output


def isString(str1):
    """
    Check if input is a string
    """
    pattern = re.compile(r'[A-Za-z]')
    res = re.findall(pattern, str(str1))
    if len(res):
        return True
    else:
        return False


def cal_sentence_count(Input):
    """
    Calculate number of sentences in a document
    """
    # combine topic and content to the full doc
    topic = Input['topic']
    content = Input['content']
    proc = topic + ' ' + content

    # replace return and new line characters
    proc = content.replace("\n", " ")
    proc = proc.replace("\r", "")
    proc = TextBlob(proc)

    # iterate through sentences
    count = 0
    for sentence in proc.sentences:
        if not sentence.strip() or not isString(sentence.strip()):
            continue
        count = count + 1

    return count

def cal_paragraph_count(Input):
    """
    Calculate number of paragraphs in a document
    """
    # combine topic and content to the full doc
    topic = Input['topic']
    content = Input['content']
    proc = topic + ' ' + content

    # split into paragraphs
    proc = proc.split('\n\n')

    # get count of paragraphs
    count = len(proc)

    return count

def cal_stdev_per_sentence(Input):
    """
    Calculate standard deviation of words per sentence
    """
    # combine topic and content to the full doc
    topic = Input['topic']
    content = Input['content']
    proc = topic + ' ' + content

    # tokenize to sentences
    proc = nltk.sent_tokenize(proc)
    
    # create list of word counts per sentence
    word_counts = [len(nltk.word_tokenize(sentence)) for sentence in proc]

    # calculate standard deviation
    std = np.std(word_counts)

    return std

def cal_stdev_per_paragraph(Input):
    """
    Calculate standard deviation of words per paragraph
    """
    # combine topic and content to the full doc
    topic = Input['topic']
    content = Input['content']
    proc = topic + ' ' + content

    # split into paragraphs
    proc = proc.split('\n\n')

    # iterate through paragraphs for word count
    paragraph_word_counts = []
    for paragraph in proc:
        paragraph_word_counts.append(np.sum([len(nltk.word_tokenize(sentence)) for sentence in paragraph]))

    # calculate the standard deviation
    std = np.std(paragraph_word_counts)

    return std

def cal_stdev_sent_per_paragraph(Input):
    '''
    Calculate standard deviation of sentences per paragraph
    '''
    # combine topic and content to the full doc
    topic = Input['topic']
    content = Input['content']
    proc = topic + ' ' + content

    # split into paragraphs
    proc = proc.split('\n\n')
    sent_counts = [len(nltk.sent_tokenize(paragraph)) for paragraph in proc]

    # calculate standard deviation
    std = np.std(sent_counts)
    return std

def cal_sentiment(Input):
    '''
    Calculate the sentiment with text blob
    '''
    my_blob = TextBlob(str(Input))
    sentiment = str(my_blob.sentiment.polarity)

    return sentiment


def cal_jaccard_coef(Input):
    '''
    Calculate the jaccard coefficient of topic and content
    '''
    topic = Input['topic_processed']
    content = Input['content_processed']

    # List the unique words in a document
    words_topic = set(topic.lower().split())
    words_content = set(content.lower().split())
    
    intersection = words_topic.intersection(words_content)

    union = words_topic.union(words_content)

    jaccard_coefficient = float(len(intersection))/len(union)

    return jaccard_coefficient


def cal_cosine_similarity(Input):
    '''
    calculate the cosine similarity between topic and content of input
    '''
    topic = Input['topic']
    content = Input['content']

    data = [topic, content]

    # vectorize the data
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(data)

    # calculate cosine similarity
    cosine_similarity_matrix = cosine_similarity(vector_matrix)

    return cosine_similarity_matrix[0][1]


def cal_cosine_similarity_on_processed_data(Input):
    '''
    Calculate the cosine similarity between topic and content of pre-processed input
    '''
    topic = Input['topic_processed']
    content = Input['content_processed']

    data = [topic, content]

    # vectorize the data
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(data)

    # calculate cosine similarity
    cosine_similarity_matrix = cosine_similarity(vector_matrix)

    return cosine_similarity_matrix[0][1]


def cal_cosine_similarity_spacy(Input):
    '''
    Calculate cosine similarity of topic and content using spacy
    '''
    topic = nlp(Input['topic'])
    content = nlp(Input['content'])

    return (content.similarity(topic))


def cal_cosine_similarity_spacy_on_processed_data(Input):
    '''
    Calculate cosine similarity of pre-processed topic and content using spacy
    '''
    topic = Input['topic_processed']
    content = Input['content_processed']

    topic = nlp(topic)
    content = nlp(content)

    return (content.similarity(topic))


def cal_wmd(Input):
    '''
    Calculate the word movers distance between topic and content of input
    '''
    topic = Input['topic_processed']
    content = Input['content_processed']

    global model
    wmd = model.wmdistance(topic, content)

    return wmd


def cal_wmd_split(Input):
    '''
    Calculate the word movers distance between topic and content of input as lists
    '''
    topic = Input['topic_processed']
    content = Input['content_processed']

    wmd = model.wmdistance(topic.split(), content.split())

    return wmd

def cal_wmd_domain(Input):
    '''
    Calculate word mover's distance using domain model word2vec
    '''
    topic = Input['topic_processed']
    content = Input['content_processed']

    global model_domain
    wmd = model_domain.wv.wmdistance(topic.split(), content.split())

    return wmd


def cal_wmd_cyber(Input):
    '''
    Calculate word mover's distance using cyber word2vec
    '''
    topic = Input['topic_processed']
    content = Input['content_processed']

    global model_cyber
    wmd = model_cyber.wv.wmdistance(topic, content)

    return wmd

def cal_wmd_cyber_split(Input):
    '''
    Calculate word mover's distance using cyber word2vec as lists
    '''
    topic = Input['topic_processed']
    content = Input['content_processed']

    global model_cyber
    wmd = model_cyber.wv.wmdistance(topic.split(), content.split())

    return wmd

def cosine(u, v):
    '''
    Calculate cosine similarity of provided values
    '''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def cal_average_cosine_similarity_sentencebert(Input):
    '''
    Calculate average cosine similarity with bert
    '''
    topic = Input['topic']
    content = Input['content']

    global sbert_model
    topic_vec = sbert_model.encode([topic])[0]

    proc = content.replace("\n", " ")
    proc = proc.replace("\r", "")
    proc = TextBlob(proc)

    cosine_similarity_list = []
    for sentence in proc.sentences:
        if not sentence.strip() or not isString(sentence.strip()):
            continue
        sim = cosine(topic_vec, sbert_model.encode([sentence])[0])
        cosine_similarity_list.append(sim)

    output = mean(cosine_similarity_list)
    return output
  
def flesch_kincaid(Input):
    '''
    Calculate flesch-kincaid readability
    '''
    # combine topic and content to full doc
    content = Input['topic'] + ' ' +  Input['content']

    # check the length is long enough for readbility score
    if len(content.split()) <100:
        return None
    
    # generate readbility score
    r = Readability(content)
    fk = r.flesch_kincaid()

    return fk.score

def flesch_reading_ease(Input):
    '''
    Calculate flesch reading ease
    '''
    # combine topic and content to full doc
    content = Input['topic']  + ' ' + Input['content']

    # check the length is long enough for readbility score
    if len(content.split()) <100:
        return None
    
    # generate readbility score
    r = Readability(content)
    flesch = r.flesch()

    return flesch.score

def gunning_fog(Input):
    '''
    Calculate gunning fog readability
    '''
    # combine topic and content to full doc
    content = Input['topic']  + ' ' + Input['content']

    # check the length is long enough for readbility score
    if len(content.split()) <100:
        return None
    
    # generate readbility score
    r = Readability(content)
    gf = r.gunning_fog()
    
    return gf.score

def automatic_readability(Input):
    '''
    Calculate automatic readability index readbility
    '''
    # combine topic and content to full doc
    content = Input['topic']  + ' ' + Input['content']

    # check the length is long enough for readbility score
    if len(content.split()) <100:
        return None
    
    # generate readbility score
    r = Readability(content)
    ari = r.ari()

    return ari.score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ####################################################################
    # Parse command line
    ####################################################################

    # Default Dirs
    parser.add_argument('--dataDir', type=str, default='./dataset/', help='intput Corpus folder')

    # input & output
    parser.add_argument('--input', type=str, default='CTI_long.xlsx', help='input file name')
    parser.add_argument('--output', type=str, default='dataset_long_with_features.xlsx', help='output file name')

    m_args = parser.parse_args()
    main(m_args)
