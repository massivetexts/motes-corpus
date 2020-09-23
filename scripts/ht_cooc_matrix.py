import argparse
from motes_corpus.hathibook import HathiCorpus
import motes_corpus.modeling as modeling
from htrc_features import utils
from scipy import sparse
import os
import time
import random
from gensim.corpora import dictionary

def main():
    parser = argparse.ArgumentParser(description='Take a list of HTIDs and process a coocurrence matrix.')
    
    parser.add_argument('--outdir', default='/data/motes/ht_coocs', type=str,
                        help='Dir to save the resulting sparse matrix npz')
    parser.add_argument('--window_size', '-w', type=int, default=100, help="Size of context window. Since HathiTrust"
                        "Books are shuffled, a small window is unnecessary")
    parser.add_argument('--trunc_doc_at', '-t', type=int, default=500, help="Truncate a page word list at this number of words.")
    parser.add_argument('htids', nargs='+', type=str, help="HTIDs to process.")
    args = parser.parse_args()
    
    htid2path = lambda x: '/data/extracted-features-parquet-stubby/{}.tokens.parquet'.format(utils.id_to_stubbytree(x))
    book_paths = ([htid2path(htid) for htid in args.htids])
    
    dict_path = '/data/motes/gigaword_300_dict.txt'
    model_dict = dictionary.Dictionary.load_from_text(dict_path)
    
    fname = "ht_cooc_{}_{}.npz".format(time.strftime('%m-%d'), random.randint(0, 10**6))
    outf = os.path.join(args.outdir, fname)
    
    htcorpus = HathiCorpus(book_paths)
    
    cooc = modeling.train_coocurrence_matrix(htcorpus.tokens(),
                                             model_dict,
                                             window_size=args.window_size,
                                             print_every=1000,
                                             size_weighted=True,
                                             trunc_doc_at=args.trunc_doc_at,
                                             context_weighted=False)
    
    sparse.save_npz(outf, cooc.tocoo())
    
if __name__ == '__main__':
    main()