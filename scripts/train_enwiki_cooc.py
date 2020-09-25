from gensim.corpora import WikiCorpus, dictionary
from motes_corpus import modeling
from scipy import sparse
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outpath", type=str, help="Where to save the cooc matrix")
    parser.add_argument("--skip_count", "-n", type=int, help="Number of docs to skip.", default=0)
    parser.add_argument("--max-docs", "-m", type=int, help="Max docs to process. Useful for early stopping before all the server's memory is used.", default=666666)
    parser.add_argument("--prune-below", default=.21)
    parser.add_argument("--prune-every", default=50000)
    parser.add_argument("--fold-every", default=1000)
    args = parser.parse_args()
    
    dict_path = '/data/motes/gigaword_300_dict.txt'
    model_dict = dictionary.Dictionary.load_from_text(dict_path)

    wiki = WikiCorpus('/data/motes/enwiki/enwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary=model_dict)

    def skip_texts(n, max=False):
        '''Resume processing after a given point'''
        for i, text in enumerate(wiki.get_texts()):
            if i < n:
                if i % 20000 == 0:
                    print(i, 'skipped')
                continue
            else:
                yield text
            if max and ((i-n) >= max):
                return

    wikicooc = modeling.train_coocurrence_matrix(skip_texts(args.skip_count, args.max_docs), model_dict,
                                                 window_size=10, print_every=5000, prune_every=args.prune_every, 
                                                 prune_below=args.prune_below, fold_every=args.fold_every)
    print("Building complete. Saving matrix")
    sparse.save_npz(args.outpath, wikicooc)
    
if __name__ == "__main__":
    main()