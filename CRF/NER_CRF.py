# %% [markdown]
# ### Conditional Random Fields (CRFs)

# %%
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn_crfsuite
import os
import random
import nltk
import re
from collections import Counter, defaultdict, namedtuple, OrderedDict
from itertools import chain
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from io import BytesIO
from itertools import chain
from nltk.corpus import stopwords

# %%
Test_Sentence = namedtuple("Sentence", "words")

def getMeTestSentences(data):
    sentences = []
    for key in data:
        sentence = []
        for val in zip(data[key].words):
            sentence.append(val)
        sentences.append(sentence)
    return sentences

def read_test_data(filename):
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
        index = 1
        a = OrderedDict()
        for s in sentence_lines:
            temp = []
            for l in s:
                temp.append(l.strip().split("\t")[1:])   
            temp2 = []
            for val in temp:
                if len(val) == 1:
                    temp2.append(val[0])
                          
            a[index] = Test_Sentence(tuple(temp2))
            index += 1
        return a
        
class TestDataset(namedtuple("_TDataset", "sentences keys vocab X N")):
    def __new__(cls, tagfile, datafile):
        sentences = read_test_data(datafile)
        keys = tuple(sentences.keys())
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        word_sequences = tuple([sentences[k].words for k in keys])
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences,N)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())

# %%
Sentence = namedtuple("Sentence", "words tags")

def getMeSentences(data):
    sentences = []
    for key in data:
        sentence = []
        for val in zip(data[key].words,data[key].tags):
            sentence.append(val)
        sentences.append(sentence)
    return sentences

def read_data(filename):
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
        index = 1
        a = OrderedDict()
        for s in sentence_lines:
            temp = []
            for l in s:
                temp.append(l.strip().split("\t")[1:])
            
            temp2 = []
            temp3 = []
            for val in temp:
                if len(val) == 2:
                    temp2.append(val[0])
                    temp3.append(val[1])
                          
            a[index] = Sentence(tuple(temp2),tuple(temp3))
            index += 1
        return a

def read_tags(filename):
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)

class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())

class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, tagfile, datafile, train_test_split=0.8, seed=None):
        tagset = read_tags(tagfile)
        sentences = read_data(datafile)
        keys = tuple(sentences.keys())
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        
        _keys = list(keys)
        if seed is not None: random.seed(seed)
        random.shuffle(_keys)
        split = int(train_test_split * len(_keys))
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())

# %%
data = Dataset("tags-universal.txt", "S21-gene-train.txt", train_test_split=0.8)

# %%
classes = list(data.tagset)

# %%
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# %%
def iscamelcase(string):
    non_alpha = [i for i in string if not i.isalpha()]
    substrings= string.translate({ord(i): ' ' for i in non_alpha}).split(' ')
    for string in substrings:
        if not all(char.isupper() for char in string):
            for idx,i in enumerate(string):
                if i.isupper() and idx > 0:
                    return True
    return False

# %% [markdown]
# #### Features extraction
# 
# 

# %%
def word2features(sent, i):
    word = sent[i][0]    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isStopword()': word in stop_words,
        'word.isalnum()': word.isalnum(),
        'word.endWithASE()': word.lower().endswith('ase'),
        'word.endWithIN()': word.lower().endswith('in'),
        'word.logWordLength': np.log(len(word)),
        'word.isCamelCase': iscamelcase(word),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isStopword()': word1 in stop_words,
            '-1:word.isalnum()': word1.isalnum(),
            '-1:word.endWithASE()': word1.lower().endswith('ase'),
            '-1:word.endWithIN()': word1.lower().endswith('in'),
            '-1:word.logWordLength': np.log(len(word1)),
            'isPreviousWordDash()': True if word1 == '-' else False,
            'isPreviousBracket()': True if word1 in ['(','['] else False
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isStopword()': word1 in stop_words,
            '+1:word.isalnum()': word1.isalnum(),
            '+1:word.endWithASE()': word1.lower().endswith('ase'),
            '+1:word.endWithIN()': word1.lower().endswith('in'),
            '+1:word.logWordLength': np.log(len(word1)),
            'isNextWordDash()': True if word1 == '-' else False,
            'isNextBracket()': True if word1 in [')',']'] else False

        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token,label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

# %%
len(data.sentences)

# %%
X_train = [sent2features(s) for s in getMeSentences(data.sentences)]
# X_test = [sent2features(s) for s in getMeSentences(data.testing_set.sentences)]
y_train = [sent2labels(s) for s in getMeSentences(data.sentences)]
# y_test = [sent2labels(s) for s in getMeSentences(data.testing_set.sentences)]

# %%
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# %%
y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=classes)

# %%
print(metrics.flat_classification_report(y_test, y_pred, labels = classes))

# %% [markdown]
# ## Parameter Optimisation for CRF using RandomisedSearchCV

# %%
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=classes)

rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,  
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(X_train, y_train)

# %%
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

# %%
crf = rs.best_estimator_
# y_pred = crf.predict(X_test)
# print(metrics.flat_classification_report(y_test, y_pred, labels=classes))

# %% [markdown]
# ### Printing Stats for Transitions

# %%
from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common())

# %%
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(3))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-3:])

# %% [markdown]
# ### ELI5 to visualise weights

# %%
import eli5


eli5.show_weights(crf, top=10)

# %%
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)
crf.fit(X_train, y_train)
eli5.show_weights(crf, top=10)

# %%
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)
crf.fit(X_train, y_train);
eli5.show_weights(crf, top=5, show=['transition_features'])

# %%
eli5.show_weights(crf, top=10, targets=['O', 'B', 'I'])

# %%
eli5.show_weights(crf, top=10, feature_re='^is',
                  horizontal_layout=False, show=['targets'])

# %%
eli5.show_weights(crf, top=10, feature_re='^word.is',
                  horizontal_layout=False, show=['targets'])

# %% [markdown]
# ## Writing To Files

# %%
with open('yoursystemoutput.txt', 'w') as f:
    k = 0
    for key in data.testing_set.sentences:
        for i,val in enumerate(zip(data.testing_set.sentences[key].words,y_pred[k])):
            f.write("\t".join([str(i+1),val[0],val[1]]) + "\n")
        k += 1
        f.write("\n")

with open('goldstandardfile.txt', 'w') as f:
    for key in data.testing_set.sentences:
        for i,val in enumerate(zip(data.testing_set.sentences[key].words,data.testing_set.sentences[key].tags)):
            f.write("\t".join([str(i+1),val[0],val[1]]) + "\n")
        f.write("\n")

# %%
test_data = TestDataset("tags-universal.txt", "F21-gene-test.txt")

X_testFinal = [sent2features(s) for s in getMeTestSentences(test_data.sentences)]


# %%

y_predTestFinal = crf.predict(X_testFinal)

with open('output_sumbission_paramOptimised.txt', 'w') as f:
    k = 0
    for key in test_data.sentences:
        for i,val in enumerate(zip(test_data.sentences[key].words,y_predTestFinal[k])):
            f.write("\t".join([str(i+1),val[0],val[1]]) + "\n")
        k += 1
        f.write("\n")


