from music21 import *
from compmusic import dunya
import os
import re
import copy
import csv
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np


def count_letters(x):
        counter = 0
        for let in x:
            if let!='-' and let!='#':
                counter+=1
        return counter


# Load the csv file with the database of centones regarding the tab'
data={}
with open('Centones.csv') as csvfile:
    csv_file = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_file:
        row2=row[1:]
        for r in row2:
            if r!='':
                if row[0] not in data:
                    data[row[0]]=[r]
                else:
                    data[row[0]].append(r)
                    
data = {k: [x.replace(" ","") for x in v] for k, v in data.items()} # Delete spaces in the centones and make dictionary





path = '/home/miguelgc96/Desktop'
fileName = 'a451a7fc-c53f-462a-b3fc-4377bb588105' + '.musicxml'
fn = os.path.join(path, fileName)

s = converter.parse(fn)
p = s.parts[0]
notes = p.flat.notes.stream() # These are all the notes of the whole piece
total_results3=[]
for n in notes:
    interval = None
    if (n.lyric == 's.muassa‘') or (n.lyric == 's.mahzūz') or (n.lyric == 's.inṣirāf'):
        interval=p.getElementsByOffset(n.offset, n.offset+20,
                              mustBeginInSpan=False,
                              includeElementsThatEndAtStart=False).stream()
        print(n.offset,n.offset+10,len(interval))
    elif (n.lyric == 'e.muassa‘') or (n.lyric == 'e.mahzūz') or (n.lyric == 'e.inṣirāf'):
        interval=p.getElementsByOffset(n.offset, n.offset-20,
                              mustBeginInSpan=False,
                              includeElementsThatEndAtStart=False).stream()
        print(n.offset,n.offset+10,len(interval))
        
    if interval != None:
        notes_interval = interval.flat.notes.stream()
        for d in data: # For each tab' we count the number of centones
            ncentones3=[]
            for centon in data[d]: # count the number of apperances of every centon in the set of centones of the tab'
                clen = count_letters(centon)
                numbercentones3 = 0
                for i in range(len(notes_interval[:-clen+1])): # Check all the combinations depending on the number of notes of the
                    buffer = []                       # centon and if matches-> numbercentones+=1
                    for j in range(clen):
                        buffer.append(notes_interval[i+j])
                    phrase = ''
                    for n in buffer:
                        phrase += n.name
                    if phrase == centon:
                        numbercentones += 1
                ncentones3.append(numbercentones3) # Acumulate the number of every kind of centones in a list
            total_results3.append([d,sum(ncentones3)]) # Store in a list of tuples [(name of tab', appearances of characteristics centones)]

max_centon=max(x[1] for x in total_results3)
total_results_coef3= [[x[0],x[1]/max_centon] for x in total_results3]
