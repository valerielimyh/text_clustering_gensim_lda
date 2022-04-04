# NLP: Text clustering
 
## Objective

Group the news in the provided news dataset to various clusters. 

## Methodology
- I conducted 3 expts with the following pipeline
### Expt1: Remove english stopwords and words that appear >60% in document, then perform lemmatization and build baseline model
1. Text preprocessing
2. Create count vectorizer to convert text into vectors
3. Create a baseline model using Gensim’s LDA to cluster each document 
4. Visualize the cluster using word cloud (to display the top N words)
5. Evaluate the model using Coherence score

### Expt2: Remove english stopwords and words that appear >30% in document, then perform lemmatization and build 2nd model
- around 24% improvement in Coherence Score in expt 2 c.f. expt 1
### Expt3: Perform hyperparameter tuning to get optimize hyperparameters (alpha, beta, number of topics), then apply the best combination of hyperparameters to create the optimum LDA model
- using the best hyperparameters from hyperparameter tuning improved expt2's result by 10%
Topics: 
Topic 0: stock market
Topic 1: music, gaming
Topic 2: sports
Topic 3: politics
Topic 4: sports
Topic 5: gadgets
Topic 6: government

## Next steps
- Explore whether TF-IDF vectorizer results in more clearly defined clusters
- Explore contextual based language models (e.g. transformers embeddings) that can capture the contextual meaning of each article, instead of mere frequency count/tf-idf
    - Nevertheless, simple count vectorizer serves as a baseline to check if contextual embeddings improves model performance.
    - For production purpose, can consider DistilBert embeddings as it is a more lightweight model with comparable performance as BERT. Model distillation reduces latency
- However, the maximum length of tokens for BERT is 512. As such, any words after that would be truncated. Given that we've some long articles, other varients of transformers e.g. Longformer could be explored to generate the embeddings for longer articles.

## To view & run the notebook
- create and activate conda env
```
$ conda create -n sandbox_conda python=3.7
$ conda activate sandbox_conda
$ jupyter lab
```

