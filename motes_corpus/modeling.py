from scipy import sparse
import time
import gensim
import numpy as np
import pandas as pd
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models import KeyedVectors
from collections import OrderedDict

class MOTESCorpus(object):
    ''' Generic corpus reader. Needs a class-level DocClass set, which takes a document path'''
        
    def __init__(self, subtitle_paths):
        self.paths = subtitle_paths
        self._stats = None
        
        nlp = English()
        self.tokenizer = nlp.tokenizer
        
    def docs(self):
        for path in self.paths:
            yield self.DocClass(path)
            
    def texts(self):
        for doc in self.docs():
            try:
                text = doc.text
            except:
                text = ''
            if type(text) is str:
                yield text
            elif type(text) is list:
                for subunit in text:
                    yield subunit
            
    def tokens(self, norm=True):
        if self.DocClass.pre_tokenized:
            for doc in self.docs():
                tokens = doc.tokens()
                if len(tokens) > 0:
                    if type(tokens[0]) is list:
                        for subunit in tokens:
                            yield subunit
                    else:
                        yield tokens
        else:
            for doc in self.tokenizer.pipe(self.texts()):
                if norm:
                    yield [token.norm_ for token in doc]
                else:
                    yield [token.text for token in doc]
                
    def stats(self):
        '''
        Iterate through the corpus and count words and docs
        '''
        if not self._stats:
            stats = dict(words=0, docs=0)
            unique_words = set()
            for text in self.texts():
                words = text.split()
                unique_words = unique_words.union(words)
                stats['docs'] += 1
                stats['words'] += len(words)
            stats['types'] = len(unique_words)
            self._stats = stats
        return self._stats

def weigh_contexts(tokenids, left=True):
    '''
    Calculate weights for context words
    '''
    n = len(tokenids)
    weights = [i/n for i in range(1, 1+n)]
    if not left:
        weights = weights[::-1]
    return list(zip(tokenids, weights))

def train_coocurrence_matrix(doc_word_iter, model_dict, window_size=3, context_weighted=True,
                             print_every=1000, prune_every=False, prune_below=False, trunc_doc_at=False, size_weighted=False,
                            fold_every=100):
    '''
    doc_word_iter: Iterator that returns raw lists of words with each step
    window_size: Size of context on each side of target
    trunc_doc_at: Specify number of words to shorted the input document to. Potentially useful for 
        BOW training
    size_weighted: If True, weights will be divided by size of the document. This may help account for doc
        size issues when using bags-of-words, since there's no real context.
    fold_every: How often should an intermediate coocurrence matrix be created and folded into the main matrix (every n documents).
        This is because the sparse matrices are constructed in COO, which is fast but requires more memory - we're collecting
        triples of row,col,weight for each coocurrences, without summing them until the folding.
    '''
    start = time.time()
    n_words = len(model_dict)
    cooc = sparse.coo_matrix((n_words, n_words), dtype=np.float)

    data, row, col = [], [], []

    try:
        for i, words in enumerate(doc_word_iter):
            if trunc_doc_at:
                words=words[:trunc_doc_at]

            # Convert words to token ids
            token_ids = model_dict.doc2idx(words)

            # Process context window and add to coocurrence matrix
            for target_i, target_id in enumerate(token_ids):
                if target_id == -1:
                    continue
                left_window = max(0, target_i - window_size)
                left_tokenids = token_ids[left_window:target_i]

                right_window = min(len(token_ids), target_i + window_size)
                right_tokenids = token_ids[target_i+1:right_window+1]

                if context_weighted:
                    all_weighted = weigh_contexts(left_tokenids) + weigh_contexts(right_tokenids, left=False)
                else:
                    all_weighted = [(context_id,1) for context_id in left_tokenids+right_tokenids]

                if size_weighted:
                    all_weighted = [(c,(w/len(all_weighted))) for c,w in all_weighted]

                for context_id, weight in all_weighted:
                    if target_id > 0 and context_id > 0:
                        row.append(target_id)
                        col.append(context_id)
                        data.append(weight)
                    
            if i % fold_every == 0:
                cooc_intermediate = sparse.coo_matrix((data, (row, col)), shape=(n_words, n_words))
                cooc_intermediate.sum_duplicates()
                cooc += cooc_intermediate
                
                data, row, col, cooc_intermediate = [], [], [], None
                
            if i % print_every == 0:
                progress = time.time() - start
                print("Docs processed: {}\t time: {}s\t docs/second: {}".format(i, int(progress), int(i/progress)))
                
            if prune_every and i % prune_every == 0 and i > 0:
                if not prune_below:
                    print('No pruning limit set with prune_below!')
                else:
                    print('pruning infrequent coocurrences')
                    before_size = len(cooc.data)
                    cooc.data[cooc.data < prune_below] = 0
                    cooc.eliminate_zeros()
                    after_size = len(cooc.data)
                    print('Reduced size from to {} to {} ({}%)'.format(before_size, after_size, int(100*(after_size/before_size))))
    except KeyboardInterrupt:
        print("Manually stopping training and return coccurrence matrix ")
        progress = time.time() - start
        print("Docs processed: {}\t time: {}s\t docs/second: {}".format(i, int(progress), int(i/progress)))

    cooc_intermediate = sparse.coo_matrix((data, (row, col)), shape=(n_words, n_words))
    cooc_intermediate.sum_duplicates()
    cooc += cooc_intermediate
    data, row, col, cooc_intermediate = [], [], [], None
    
    if prune_below:
        print('Final prune')
        cooc.data[cooc.data < prune_below] = 0
        cooc.eliminate_zeros()
        
    progress = time.time() - start
    print("Final docs processed: {}\t time: {}s\t docs/second: {}".format(i, int(progress), int(i/progress)))
    
    return cooc


def merge_sparse_matrices(paths, weighted=False):
    '''if weighted, expects path, weight tuples'''
    for i, path in enumerate(paths):
        if weighted:
            path, weight = path
            partial_mat = sparse.load_npz(path) * weight
        else:
            partial_mat = sparse.load_npz(path)
        if i == 0:
            full_mat = partial_mat
        else:
            full_mat += partial_mat
    return full_mat


def glove_to_keyedvectors(glove):
    '''
    Convert a GloVe model trained by Maciej Kula's Glove Python library, to a KeyedVectors
    representation in Gensim.
    The latter is a bit easier to work with and save.
    '''
    kv =  gensim.models.keyedvectors.KeyedVectors(len(glove.dictionary))
    kv.vectors = glove.word_vectors
    # Add vocab
    vocab = OrderedDict()
    for word, idx in glove.dictionary.items():
        tmp = gensim.models.keyedvectors.Vocab()
        tmp.index = idx
        vocab[word] = tmp
    kv.vocab = vocab
    kv.index2word = list(vocab.keys())
    return kv


class ModelCompare(object):
    
    def __init__(self, target_model_path, baseline_model_path, include_list=False):
        self.target = KeyedVectors.load(target_model_path)
        self.baseline = KeyedVectors.load(baseline_model_path)
        self._scores = {}
        
        if include_list:
            self.truncate_models(include_list)
    
    def truncate_models(self, include_list):
        self.target = self._truncate_keyedvectors(self.target, include_list)
        self.baseline = self._truncate_keyedvectors(self.baseline, include_list)
        
    def compare_nns(self, q, k = 100, ordered=True):
        ''' If ordered, then output lists rather than sets. Score is from Gonan et al 2020 (when k=1000)'''
        target_words = [w for w, sim in self.target.most_similar(q, topn=k)]
        baseline_words = [w for w, sim in self.baseline.most_similar(q, topn=k)]

        if ordered:
            intersect = [word for word in baseline_words if word in target_words]
            a = [word for word in baseline_words if word not in target_words]
            b = [word for word in target_words if word not in baseline_words]
        else:
            intersect = set(baseline_words).intersection(target_words)
            a = set(baseline_words).difference(target_words)
            b = set(target_words).difference(baseline_words)
        return dict(general=a, child=b, overlap=intersect, score=len(intersect)/k)
    
    def print_compare_nns(self, q, k=100):
        ''' Prettier print for word comparisons'''
        nns = self.compare_nns(q, k)
        print("Words similar to '{}'".format(q).center(50, '='))
        print('Just General Language\n', ", ".join(nns['general']))
        print('\nJust Children\'s Language\n', ", ".join(nns['child']))
        print('\nOverlaps\n', ", ".join(nns['overlap']))
        
    def compare_nns_display_table(self, list_of_qs, k=20):
        ''' Another view for comparing nns visually.'''
        rows = []
        for word in list_of_qs:
            nns = self.compare_nns(word, k)
            nns_text = "GENERAL: {}\nCHILD: {}\nOVERLAPS: {}"
            nns_text = nns_text.format(", ".join(nns['general']),
                                       ", ".join(nns['child']),
                                       ", ".join(nns['overlap']) if len(nns['overlap']) else '-'
                                      )
            rows.append({"word": word, "Nearest Neighbours": nns_text})
        return pd.DataFrame(rows).set_index('word')

    def get_intersect_score(self, q, k=1000):
        return self.compare_nns(q, k, ordered=False)['score']
    
    def _truncate_keyedvectors(self, kv, include_list):
        ''' Truncate a KeyedVectors word embedding model to a given vocabulary'''
        new_vocab = dict()
        include_dims = []

        for i, word in enumerate(include_list):
            include_dims.append(kv.key_to_index[word])
            new_vocab[word] = i

        new_kv = gensim.models.keyedvectors.KeyedVectors(len(new_vocab))
        new_kv.key_to_index = new_vocab
        new_kv.index_to_key = list(new_vocab.keys())
        new_kv.vectors = kv.vectors[include_dims]
        return new_kv
    
    def scores(self, k=1000, break_at=False, force=False):
        ''' Score words. Caches, so if you change k, remember to force recalculation'''
        if (len(self._scores) == len(self.target)) and not force:
            return pd.Series(self._scores)
        
        for i, q in enumerate(self.target.key_to_index.keys()):
            if (q not in self._scores) or force:
                score = self.get_intersect_score(q, k=k)
                self._scores[q] = score
            if (break_at) and (i > break_at):
                break

        return pd.Series(self._scores)

def word_stopping(x):
    ''' Remove certain words '''
    if not x.isalpha():
        return False
    elif len(x) == 1:
        return False
    elif x in STOP_WORDS:
        return False
    else:
        return True