'''双向 GRU 和 LSTM处理模块'''
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import keras
print(keras.__version__)

from keras.utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense, GRU, Bidirectional, Input
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.optimizers import Adam

# set seeds for reproducability
import tensorflow
from numpy.random import seed
tensorflow.random.set_seed(2)
keras.utils.set_random_seed(42)
seed(1)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from keras.layers import Embedding
import warnings
from gensim.models import KeyedVectors, Word2Vec
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


def data():
    '''
    Function to read the data
    '''
    print("###Prepare Dataset###")
    df = pd.read_excel('./dataset/dataset_long_with_features.xlsx', engine="openpyxl", sheet_name="Sheet1", header=0)
    input_rows = df.shape[0]
    print("Read input file. Rows: " + str(input_rows))

    # combine topic and content to full text
    df['combined_processed'] = df['topic_processed'] + ' ' + df['content_processed']
    
    #convert labels
    tqdm.pandas(desc="Converting Labels")
    df['label'] = df['label'].progress_map(lambda x: 1 if x == 'Real' else 0)

    # select required columns
    df = df[['combined_processed', 'label']]
    return df


# embeddings = Word2Vec.load('./model/Domain-Word2vec.model.gz').wv
embeddings = Word2Vec.load('./model/1million.word2vec.model').wv
# embeddings = KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin.gz', binary=True)

def create_embedding_matrix(embedding_model, word_index, embedding_dim):
    '''
    Create embedding matrix to convert between tokized values and pre-trained embeddings
    '''
    # Initialize an embedding matrix with zeros
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    # find corresponding indices
    for word, i in word_index.items():
        if word in embedding_model:
            embedding_matrix[i] = embedding_model[word]

    return embedding_matrix

tokenizer = Tokenizer()

def get_tokens(corpus):
    '''
    Method to tokenize text data
    '''
    # tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    # convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        input_sequences.append(tokenizer.texts_to_sequences([line])[0])
    return input_sequences, total_words

def generate_padded_sequences(input_sequences):
    '''
    Method to pad input into same length sequences
    '''
    # get the max length
    max_sequence_len = max([len(x) for x in input_sequences])

    # pad with zeros to max length
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))
    return input_sequences, max_sequence_len

def InstantiateModel(total_words, max_sequence_len):
    '''
    Method to create the model for learning
    '''
    # create input layer
    input_layer = Input(shape =(None,), dtype="int64")
    
    # create embedding matrix to convert between tokenized values and embeddings
    embedding_matrix = create_embedding_matrix(embeddings, tokenizer.word_index, 100)
    embedding = Embedding(
                        input_dim=len(tokenizer.word_index) + 1,
                        output_dim=100,
                        mask_zero = True,
                        weights=[embedding_matrix],
                        input_length=max_sequence_len,
                        trainable=False)(input_layer)
    
    # embedding = Embedding(input_dim=total_words+1, output_dim=75, input_length=max_sequence_len, mask_zero = True)(input_layer)
    # create bi-directional layers
    gru_layer = GRU(8, dropout=0.1)
    lstm_layer = LSTM(8, dropout=0.1)
    bi_gru = Bidirectional(gru_layer)(embedding)
    bi_lstm = Bidirectional(lstm_layer)(embedding)

    # concatenate layers
    merged = keras.layers.concatenate([bi_gru, bi_lstm], axis=1)

    # create linear layers
    dense = Dense(128, activation='relu')(merged)
    dense = Dense(1, activation='sigmoid')(merged)
    
    # instantiate model and compile
    model = Model(inputs=input_layer, outputs=dense)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics = ['accuracy'])
    return model

# load the data
df = data()
corpus = df['combined_processed']

# convert to tokens
inp_sequences, total_words= get_tokens(corpus)

# pad to same length
X, max_sequence_len = generate_padded_sequences(inp_sequences)

# get labels
y = df['label']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# convert data to arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
y_train = np.array(y_train)
y_test = np.array(y_test)

# load the model
model = InstantiateModel(total_words, max_sequence_len)

# set parameters
num_epochs = 500
num_examples = len(X_train)
batch_size = 128

# fit the model to training data
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, shuffle=True, steps_per_epoch=20)

# get loss values
trainLoss = history.history['loss']
testLosses = history.history['val_loss']

# get predictions for test
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.3).astype(int)

# plot the training and test loss
plt.plot(range(num_epochs), trainLoss, label='Train Loss')
plt.plot(range(num_epochs), testLosses, label='Test Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()

# Plot the confusion matrix as a heatmap
conf_matrix = confusion_matrix(y_test, y_pred_classes, normalize='pred')
sn.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=range(2), yticklabels=range(2))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
