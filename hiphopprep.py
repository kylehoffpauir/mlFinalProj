from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import string
import nltk
import itertools
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import unidecode

# Adapted from:
# https://www.kaggle.com/code/kerneler/starter-rap-lyrics-8dd83bf6-f
# https://www.kaggle.com/code/offmann/nlp-on-hiphop-lyrics
import unicodedata

def remove_accents(input_str):
    for l in input_str:
        for character in l:
            line = ""
            if character.isalnum() :
                line += character
        input_str += re.sub("[\w\-\s]","",line)
    return input_str
data = {}
frames = []
df = pd.DataFrame()
for dirname, _, filenames in os.walk('hiphop/'):
    for filename in filenames:
        rapper = filename.split('_')[0]
        with open(os.path.join(dirname, filename), encoding='utf-8') as f:
            content = remove_accents(f.read())
            content = content.splitlines()
            for i in range(len(content)-2, -1, -1):
                if content[i].isascii():
                    content[i] = (content[i] + ' ' + content.pop(i+1)).strip()
            verse = content[1::2]
            if verse is "":
                continue
            data['rapper'], data['verse'] = rapper, verse
            df_temp = pd.DataFrame(data)
            frames.append(df_temp)
            df = pd.concat(frames)
pd.set_option('max_colwidth', None)
print(df.head())

# Drop the duplicate verses (they appear several times in choruses)
print(len(df))
df = df.drop_duplicates(subset=['verse'])
df.dropna(inplace= True)
print(len(df))
# Let's regroup tupac1 and tupac2 lyrics as one single rapper
func = lambda x: 'Tupac' if 'Tupac' in x else x
df['rapper'] = df['rapper'].map(lambda x:func(x))

# For each rapper, how many unique verses do they have in the dataset
print(df.rapper.value_counts())



# Let's preprocess the verses

# Import nltk stopwords
stopwords = nltk.corpus.stopwords.words('english')

def preprocess_verse(verse, stopwords):

    verse = verse.lower()

    verse = verse.translate(str.maketrans('', '', string.punctuation))

    verse = verse.replace('\n\n',' ')

    # remove english stopwords
    verse = ' '.join([word for word in verse.split() if word not in stopwords])

    return verse


# bigram function for 2 words
def bigram(s):
    s1 = s.split()
    s1 = list(zip(s1[:-1], s1[1:]))
    s1 = list(map(lambda x: '_'.join(x), s1))

    return ' '.join(s1)
df['verse_preprocessed'] = df['verse'].map(lambda x:preprocess_verse(x, stopwords))
df['verse_preprocessed'] = df['verse_preprocessed'].dropna()
df['verse_bigrams'] = df['verse_preprocessed'].map(lambda x:bigram(x))
print(df.head(2))
# Let's see if we have any weird words in the dataset. Let's plot the occurrences of the words

def vocab(df, col, nb_words, stopwords):

    vocab = df[col].str.split(expand=True).stack().value_counts().head(50).to_dict()

    vocab_sw = {key:value for (key,value) in vocab.items() if key not in stopwords}

    return dict(itertools.islice(vocab_sw.items(), nb_words))


def plot_words(vocab):

    plt.rcParams['figure.figsize'] = (20, 10)
    plt.show()

    plt.xlim(0,len(vocab))
    plt.xticks(rotation=90,fontsize=14)
    plt.bar(vocab.keys(), vocab.values(), width=0.3, color='g')
    plt.show()


#plot_words(vocab(df, 'verse_preprocessed', 40, stopwords))



def word_cloud_by_rapper(rapper, col):

    df_temp = df[df.rapper==rapper]

    text = ' '.join(str(comment) for comment in df_temp[col])

    wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color="white",max_words=100).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.rcParams['figure.figsize'] = (20, 20)
    plt.axis("off")
    plt.show()


#word_cloud_by_rapper('Earl Sweatshirt', 'verse_preprocessed')
#word_cloud_by_rapper('Pusha-T', 'verse_bigrams')

from urllib.request import urlopen

target_url = 'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt'
data = urlopen(target_url).read().decode()
curse_words = data.replace("\n",' ').split(' ')


def is_curse(verse, curse_words):
    curse = 0

    words = verse.split()

    for word in words:
        if word in curse_words:
            curse = 1

    return curse

df['is_curse'] = df['verse_preprocessed'].map(lambda x:is_curse(x, curse_words))

# Percentage of verses that contain curse words per rapper
curse_dict = {}

for rapper in df.rapper.value_counts().keys():
    curse_dict[rapper] = dict(df[df.rapper==rapper].is_curse.value_counts(normalize=True)*100)

print(curse_dict)


# Let's plot the results

curse_df = pd.DataFrame(curse_dict).T
print(curse_df.info())
# plot grouped bar chart
curse_df.sort_values(1).plot(kind="bar",figsize=(18,18))
plt.show()

# https://www.kaggle.com/code/paultimothymooney/poetry-generator-rnn-markov

from flair.models import TextClassifier
from flair.data import Sentence
sia = TextClassifier.load('en-sentiment')
def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    print(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return "pos"
    elif "NEGATIVE" in str(score):
        return "neg"
    else:
        return "neu"

df["sentiment"] = df['verse_preprocessed'].apply(flair_prediction)
df.to_csv('cleanedRap2.csv',encoding='utf-8-sig')

