# nlp libraries for text preprocessing
import re
import nltk
#Importing matplotlib for wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud 
# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel # to evaluate lda model
# set seed to ensure reproduciblity
SEED = 100

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    '''
    helper fn to remove stopwords, stemming or lemmatization
    args: 
        - string
        - stemming flag 
        - lemmatization flag
        - list of stopwords
    return:
        - list of tokenized words
    '''
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().replace("/", " ").strip())
    
    ## Tokenize (convert from string to list)
    lst_text = text.split(' ')
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    return lst_text

def create_custom_stopwords_list(common_words_threshold, n_freq_words, df, text_col):
    '''
    helper fn to create a list of additional stopwords based on how frequent the word appears across all documents
    args: 
        - word frequency threshold
        - a list of tuple of common words (word, frequency)  
        - df with the text_col 
        - text_col to filter for word
    return:
        - list of additional stopwords
    '''
    custom_stopwords_list = []
    for word in n_freq_words:
        word = word[0].strip(',')
        if len(df[df[text_col].str.contains(word)])/len(df) * 100 > common_words_threshold:
            custom_stopwords_list.append(word)
    return custom_stopwords_list

def create_dict_and_corpus(docs):
    '''
    helper fn to create dict & corpus
    args: 
        - documents
    return:
        - index to word mapping
        - corpus
    '''
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(text) for text in docs]
    return id2word, corpus

def compute_coherence_values(corpus, document, dictionary, k, a, b):
    '''
    helper fn to compute c_v coherence value to evaluate lda model
    args: 
        - corpus
        - document
        - number of topics
        - alpha
        - beta
    return:
        - coherence score
    '''
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=SEED,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=document, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

def visualize_cluster_word_cloud(model, num_words_in_wc):
    '''
    helper fn to generate word cloud for each topic in lda model
    args: 
        - lda model
        - number of words to display in word cloud
    return:
        - matplotlib figure
    '''
    for t in range(model.num_topics):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(model.show_topic(t, num_words_in_wc))))
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.show()