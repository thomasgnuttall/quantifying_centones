from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import numpy.matlib


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

def occurrences(string, sub):
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count+=1
        else:
            return count
        
def count_centones(mbid,notes_dict,nawba_centones):
    notes_name=[n for n in notes_dict[mbid]]
    notes_name = ''.join(notes_name)
    # Count centones:
    nawba_and_ncentones = []
    for d in nawba_centones: # For each tab' we count the number of centones
        ncentones = []
        for centon in nawba_centones[d]: # count the number of apperances of every centon in the set of centones of the tab'
            numbercentones = occurrences(notes_name, centon)
            ncentones.append(numbercentones) # Acumulate the number of every kind of centones in a list
        nawba_and_ncentones.append([d, sum(ncentones)]) # Store in a list of tuples [(name of tab', appearances of characteristics centones)]

    nawba_and_ncentones = [[x[0],x[1]/len(nawba_centones[x[0]])] for x in nawba_and_ncentones]
    predicted = [x[0] for x in sorted(nawba_and_ncentones, key=lambda y: -y[1])][0]
    return predicted

def look_for_knee(likely_patterns):
    """
    Compute the knee point for the first 100 most likely centones for each nawba.
    Taken from: https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1
    """
    values=[x[1] for x in likely_patterns][:100]

    #get coordinates of all the points
    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T
    #np.array([range(nPoints), values])

    # get the first point
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vecFromFirst = allCoord - firstPoint

    # To calculate the distance to the line, we split vecFromFirst into two 
    # components, one that is parallel to the line and one that is perpendicular 
    # Then, we take the norm of the part that is perpendicular to the line and 
    # get the distance.
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint

