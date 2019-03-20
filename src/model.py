from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

def get_tfidf_distributions(all_recordings):
    """
    For a list of all recordings patterns tf-df-results
    Returns:
        list, each elecment a recording, summarised as a list of (pattern, tf-idf, total_count_of_pattern)
    """
    vectorizer = TfidfVectorizer(lowercase=False, sublinear_tf=True)
    X = vectorizer.fit_transform(all_recordings)
    top_patterns = []
    sorted_vocab = sorted([(k,v) for k,v in vectorizer.vocabulary_.items()], key=lambda y: y[1])
    vocab = [x[0] for x in sorted_vocab]
    for i,x in enumerate(X):
        weights = x.toarray()[0]
        pattern_counts = Counter(all_recordings[i].split(' '))
        this_patterns = []
        for v,w in zip(vocab, weights):
            tf = float(pattern_counts[v])/len(pattern_counts)
            this_patterns.append((v, tf*w, pattern_counts[v]))
        top_patterns.append(this_patterns)
    return top_patterns


def top_n(distributions, n=5):
    """n=0 to return all"""
    if not n:
        n = len(distributions)
    return sorted(distributions, key=lambda y: y[1], reverse=True)[:n]


def get_nawba_frame(distributions, nawba, nawba_mappings, nawba_indices):
    """
    For distributions from get_tfidf_distributions return dataframe of all weightings across all nawbas
    """
    distributions = {k:v for k,v in zip(nawba_indices, distributions)}[nawba]
    frame = pd.DataFrame(distributions, columns=['pattern', 'tfidf'])
    frame['is_pattern'] = frame['pattern'].apply(lambda y: any([y in p for p in nawba_mappings[nawba]]))
    return frame


def average_tfidf(distributions, indices):
    """
    For a list of tfidf results (from model.get_tfidf_distributions)
    Return:
        Dataframe of average tf-idf across recordings of the same nawba
        ...[nawba, pattern, tf-idf, frequency]
    """
    frame = zip_nawba(distributions, indices)
    frame_grouped = frame.groupby(['index', 'pattern'])\
                         .agg({'tf-idf': 'mean', 'frequency': 'sum'})\
                         .reset_index()
    return frame_grouped


def zip_nawba(distributions, indices):
    """
    Convert distributions output to DF
    """
    zip_nawba = [
        [(n,x,y,z) for x,y,z in d] \
        for n,d in zip(indices, distributions)
    ]
    frame = pd.DataFrame(
        [y for x in zip_nawba for y in x],
        columns=['index', 'pattern', 'tf-idf', 'frequency']
    )
    return frame

def centones_counter_predictor(mbid,df,number_centones_xnawba):
	"""
    Counting the number of centones of a nawba in a score, predict the nawba.
    mbid: mbid of the required score
    df: dataframe obtained from the function zip_nawba() and the distributions of IT-IDF
    number_centones_xnawba: dictionary with the nawba and the number of centones for each nawba
    """
    sub_df = df.loc[df['index'] == mbid]
    nawba_and_centones = []
    for nawba in nawba_centones:
        centones_totales=0
        for centon in nawba_centones[nawba]:
            x = sub_df.loc[sub_df['pattern'] == centon]
            if not (x.empty): 
                centones_totales += x['frequency'].iloc[0]
        nawba_and_centones.append([nawba,centones_totales])
    nawba_and_centones = [[x[0],x[1]/number_centones_xnawba[x[0]]] for x in nawba_and_centones]
    nawba_and_centones = sorted(nawba_and_centones, key=itemgetter(1))
    predicted = nawba_and_centones[-1][0]
    return predicted

