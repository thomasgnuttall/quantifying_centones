import matplotlib.pyplot as plt

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