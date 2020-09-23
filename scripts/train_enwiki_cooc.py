from gensim.corpora import WikiCorpus, dictionary
from motes_corpus import modeling
from scipy import sparse

dict_path = '/data/motes/gigaword_300_dict.txt'
model_dict = dictionary.Dictionary.load_from_text(dict_path)

wiki = WikiCorpus('/data/motes/enwiki/enwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary=model_dict)

def skip_texts(n, max=False):
    '''Resume processing after a given point'''
    for i, text in enumerate(wiki.get_texts()):
        if i < n:
            if i % 10000 == 0:
                print(i, 'skipped')
            continue
        else:
            yield text
        if max and (i >= max):
            break

wikicooc = modeling.train_coocurrence_matrix(skip_texts(348268), model_dict,
                                             window_size=10, print_every=5000)
sparse.save_npz('/data/motes/enwiki_cooc_full3_raw.npz', wikicooc)