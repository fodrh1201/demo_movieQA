import re
import cPickle as pkl
import numpy as np
from nltk.stem.snowball import SnowballStemmer

QA_DESC_TEMPLATE = 'descriptor_cache/qa.%s/%s.npy'  # descriptor, qid
DOC_DESC_TEMPLATE = 'descriptor_cache/%s.%s/%s.npy'  # document-type, descriptor, imdb-key
TFIDF_TEMPLATE = 'descriptor_cache/tfidf/source-%s.wthr-%d.pkl'  # document-type, word-threshold
RESULTS_TEMPLATE = 'results/%s.results'  # method.document-type.parameters

re_alphanumeric = re.compile('[^a-z0-9 -]+')
re_multispace = re.compile(' +')
snowball = SnowballStemmer("english")


def restrict_to_story_type(ANS, QA):
    """Restricts the set of ground truth answer options to QAs answerable
    using this story type.
    """

    QA_qids = [qa.qid for qa in QA]
    ANS = {tak:tav for tak, tav in ANS.iteritems() if tak in QA_qids}
    return ANS


def evaluate(ANS, method, method_ans):
    """Compares predicted against ground-truth answers.
    """
    # make sure all the questions are answered
    assert sorted(method_ans.keys()) == sorted(ANS.keys()), \
                '{}, All questions in the test set not answered!' %(method)
    # mark correct
    correct = []
    for qid in ANS.keys():
        if ANS[qid] == method_ans[qid]:
            correct.append(1)
        else:
            correct.append(0)
    accuracy = 100.0 * sum(correct) / len(ANS)
    print '%40s | acc. %.2f' %(method, accuracy)
    return correct, accuracy


def load_test_set_groundtruth(fname='test.answers'):
    """Load test set ground-truth. Not to be used for the actual submissions!
    """
    with open(fname) as fid:
        test_answers = fid.readlines()
    test_answers = [ta.split(' ') for ta in test_answers if ta.strip()]
    test_answers = {ta[0]:int(ta[1]) for ta in test_answers}
    return test_answers


def normalize_alphanumeric(line):
    """Strip all punctuation, keep only alphanumerics.
    """
    line = re_alphanumeric.sub('', line)
    line = re_multispace.sub(' ', line)
    return line


def normalize_stemming(line):
    """Perform stemming on the words.
    """
    words = line.split(' ')
    words = [snowball.stem(word) for word in words]
    line = ' '.join(words)
    return line


def sliding_window_text(source_list, windowsize=5):
    # convert a list into sliding window form
    sliding_window_list = []
    for k in range(len(source_list) - windowsize + 1):
        sliding_window_list.append(source_list[k : k+windowsize])

    for k in range(len(sliding_window_list)):
        sliding_window_list[k] = ' '.join(sliding_window_list[k])

    return sliding_window_list

def load_story_feature(imdb_key, story, feature):
    """Loads story feature for movie of imdb_key."""
    if feature.startswith('tfidf'):
        # TFIDF features are saved in sparse matrix format.
        with open(DOC_DESC_TEMPLATE % (story, feature, imdb_key), 'rb') as fid:
            #try:    
            story_features = pkl.load(fid)
            #except:
            #    print imdb_key
            #    return np.zeros((1, 14253))
        return story_features.todense()
    else:
        return np.load(DOC_DESC_TEMPLATE % (story, feature, imdb_key))

def load_qa_feature(qa, feature):
    return np.load(QA_DESC_TEMPLATE % (feature, qa.qid))
