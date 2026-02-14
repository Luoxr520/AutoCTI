'''特征分析模块'''
import argparse

import pandas as pd
pd.set_option("display.max_colwidth", 200)

import matplotlib.pyplot as plt

import seaborn as sns

def main(params):
    print(params)

    # read data
    input_name = params.dataDir + params.input

    print("### Start Prepare Dataset")
    print("Input: " + input_name)

    df = pd.read_excel(input_name, engine="openpyxl", sheet_name="Sheet1", header=0)

    # remove non-feature columns
    df_proc = df.drop(['topic', 'content', 'topic_processed', 'content_processed'], axis=1)

    # change labels to ints
    df_proc['label'] = df_proc['label'].map({'Real': 1, 'Fake': 0})

    # analyse features
    feature_analyse(df_proc)


def feature_analyse(df):
    """
    Create column groups to analyse together and run data distrubution
    and density
    """
    print(df.isnull().sum())
    # Data Distribution
    plt.rc("font", size=14)
    sns.set(style="white")
    sns.set(style="whitegrid", color_codes=True)
    columns_list = []
    
    # counts
    columns_list.append(['content_words_count', 'topic_words_count', 'total_words_count', 
                         'total_sentence_count', 'total_paragraph_count'])
    
    # mean and standard deviation
    columns_list.append(['mean_words_per_sentence', 'stdev_words_per_sentence', 
                         'mean_words_per_paragraph', 'stdev_words_per_paragraph',
                         'mean_sent_per_paragraph', 'stdev_sent_per_paragraph'])
    
    # sentiment
    columns_list.append(['sentiment_topic_pd', 'sentiment_content_pd', 'jaccard_coef_pd'])
    
    # cosine similarity
    columns_list.append(['cosine_similarity_sklearn', 'cosine_similarity_sklearn_pd', 
                         'cosine_similarity_spacy', 'cosine_similarity_spacy_pd'])
    
    # word mover's distance
    columns_list.append(['wmd_google_nonsplit_pd', 'wmd_google_pd', 'wmd_domain_pd', 
                         'wmd_cyber_nonsplit_pd', 'wmd_cyber_pd'])
    
    # Readability
    columns_list.append(['flesch_reading_ease', 'flesch_kincaid_readability', 
                         'gunning_fog_readability', 'automatic_readability_index'])

    # Plot distribution
    print('Data Distribution')
    for mcols in columns_list:
        range_plot(mcols, df, m_plot)

    # Plot density
    print('Density Plot for Real and Fake Data')
    for mcols in columns_list:
        range_plot(mcols, df, m_plot_by_label)


def m_plot(col_name, input_df, i_row, i_col, i):
    """
    Plot the features as histograms
    """
    x = 100 * i_row + 10 * i_col + i
    ax = plt.subplot(x)
    ax = input_df[col_name].hist(bins=15, color='teal', alpha=0.6)
    ax.set(xlabel=col_name)

def range_plot(input_cols, input_df, fun_plot):
    """
    Plot features
    """
    num_of_row = 1
    num_of_col = len(input_cols)
    plt.figure(figsize=(20,4), dpi=80)
    i = 1
    for col in input_cols:
        fun_plot(col, input_df, num_of_row, num_of_col, i)
        i+=1
    plt.show()

def m_plot_by_label(col_name, input_df, i_row, i_col, i):
    """
    Plot the features as density
    """
    ax = plt.subplot(100 * i_row + 10 * i_col + i)
    ax = sns.kdeplot(input_df[col_name][input_df.label == 1], color="darkturquoise", shade=True)
    sns.kdeplot(input_df[col_name][input_df.label == 0], color="lightcoral", shade=True)
    plt.legend(['Real', 'Fake'])
    ax.set(xlabel=col_name)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ####################################################################
    # Parse command line
    ####################################################################

    # Default Dirs
    parser.add_argument('--dataDir', type=str, default='./dataset/', help='intput folder')

    # input & output
    parser.add_argument('--input', type=str, default='dataset_long_with_features.xlsx', help='input file name')

    m_args = parser.parse_args()
    main(m_args)










