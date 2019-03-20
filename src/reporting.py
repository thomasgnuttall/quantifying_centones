import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np 

plt.rcParams["font.family"] = "sans-serif"

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


def get_top_centones_plot(frame_grouped, nawba, nawba_centones, scores_in_nawba, height=12, width=10, n=10):
    """
    Plot top <n> centones for <nawba> in <frame_grouped>

    Bars marked green are centones in <nawba_centones>[<nawba>]
    Bars marked red are centones that are superstrings of <nawba_centones>[<nawba>]
    Bars marked blue are centones not specified in lookup tables
    
    <scores_in_nawba> is used to normalise pattern frequency
        (number of scores for this Nawba)
    """ 

    nawba_string = nawba.replace('_',' ')

    # Colour scheme
    bar_edge_colour = '#919191'
    canvas_colour = '#ffffff'
    gridline_colour = '#bcbcbc'

    standard_bar_colour = '#b3cde3'
    amin_bar_colour = '#ccebc5'
    super_amin_bar_colour = '#fbb4ae'

    this_frame = frame_grouped[(frame_grouped['index'] == nawba)]\
                 .sort_values(by='tf-idf', ascending=False)[:n]

    patterns = this_frame['pattern']
    tfidf = this_frame['tf-idf']
    frequencies = [int(x/scores_in_nawba) for x in this_frame['frequency']]
    max_tfidf = max(tfidf)

    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    # gridlines beneath other elements
    ax.yaxis.grid(True, color=gridline_colour, linestyle='dashed')
    ax.set_axisbelow(True)

    # Canvas colour
    ax.set_facecolor(canvas_colour)

    # Custom Legend
    custom_boxes = [Line2D([0], [0], color=standard_bar_colour, lw=4),
                    Line2D([0], [0], color=amin_bar_colour, lw=4),
                    Line2D([0], [0], color=super_amin_bar_colour, lw=4),
                    Line2D([0], [0], color='black', lw=4)]
    ax.legend(
        custom_boxes, 
        ['New pattern', 'Amins Pattern', 'Superstring of Amins pattern', 'Average Frequency per recording']
    )

    # Example data
    y_pos = np.arange(len(patterns))

    # Create horizontal bars
    bars = ax.barh(y_pos, tfidf, align='center', color=standard_bar_colour, edgecolor=bar_edge_colour)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(patterns)
    ax.invert_yaxis()  # labels read top-to-bottom

    # Labels
    plt.title('{}, Top Ranked Patterns'.format(nawba_string), size=18)
    plt.xlabel('Average tf-idf', size=14)
    plt.ylabel('Pattern', size=14)


    # Annotate with frequency
    for i, t in enumerate(tfidf):
        ax.text(t + max_tfidf/400, i + .25, str(frequencies[i]), color='black')

    # Colour bars as per amins centones
    for i, pat in enumerate(patterns):
        if any([x == pat for x in nawba_centones[nawba]]):
            bars[i].set_color(amin_bar_colour)
            bars[i].set_edgecolor(bar_edge_colour)
        elif any([x in pat for x in nawba_centones[nawba]]):
            bars[i].set_color(super_amin_bar_colour)
            bars[i].set_edgecolor(bar_edge_colour)

    return plt