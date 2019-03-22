import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
from collections import Counter

def get_amins_plot(frame_grouped, nawba, nawba_centones):
    """
    Plot distribution from <frame_grouped> for <nawba> that are in <nawba_centones>[<nawba>]
    """
    relevant_patterns = nawba_centones[nawba]
    this_frame = frame_grouped[(frame_grouped['index'] == nawba) & (frame_grouped['pattern'].isin(relevant_patterns))].sort_values(by='tf-idf', ascending=False)
    plt.xticks(rotation=60)
    plt.title('{}, Amins Centones'.format(nawba.replace('_',' ')))
    plt.ylabel('Average tf-idf')
    plt.xlabel('Centone')
    p = plt.bar(this_frame['pattern'], this_frame['tf-idf'])
    return plt


def get_top_centones_plot(frame_grouped, nawba, nawba_centones, n=10):
    """
    Plot top <n> centones for <nawba> in <frame_grouped>
    Bars marked green are centones in <nawba_centones>[<nawba>]
    Bars marked red are centones that are superstrings of <nawba_centones>[<nawba>]
    Bars marked blue are centones not specified in lookup tables
    """ 
    this_frame = frame_grouped[(frame_grouped['index'] == nawba)].sort_values(by='tf-idf', ascending=False)[:n]
    these_patterns = this_frame['pattern']
    plt.figure(figsize=(13,10))
    plt.xticks(rotation=90)
    plt.title('{}, Non Zero Centones'.format(nawba.replace('_',' ')))
    plt.ylabel('Average tf-idf')
    plt.xlabel('Centone')
    p = plt.bar(these_patterns, this_frame['tf-idf'])
    for i,pat in enumerate(these_patterns):
        if any([x == pat for x in nawba_centones[nawba]]):
            p[i].set_color('g')
        elif any([x in pat for x in nawba_centones[nawba]]):
            p[i].set_color('r')
    return plt


def plot_centones_per_nawba(df_predicted_nawba):
    """
    Plot in a pie plot the matching annotated and predicted nawbas through the centones search
    """
    nawba_plot=[row['annotated'] for index,row in df_predicted_nawba.iterrows() if (row['annotated'] == row['predicted'])]
		    
    print('The tab with the most number of characteristics centones and the annotations matched in: ',len(nawba_plot)*100/(df_predicted_nawba.shape[0]),'%')
		
    values = Counter(nawba_plot).values()
    keys = Counter(nawba_plot).keys()
    plt.pie([v for v in values], labels=[k for k in keys],autopct='%.1f%%')
    plt.title('Data visualization of the matched tab with the annotations')
    return plt




def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return plt

