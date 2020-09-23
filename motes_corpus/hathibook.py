import random
from htrc_features import Volume
import htrc_features
from motes_corpus.modeling import MOTESCorpus

class HathiBook(object):
    
    pre_tokenized = True 
    
    def __init__(self, path):
        self.path = path
        self.vol = Volume(path=path, format='parquet')
    
    @property
    def text(self):
        return [" ".join(t) for t in self.tokens()]

    def tokens(self, max_tokens=None):
        try:
            page_group = self.vol.tokenlist(case=False, pos=False).groupby(level='page')
        except FileNotFoundError:
            print("File doesn't exist for {}".format(self.path))
            return [[]]
        except htrc_features.parsers.MissingDataError:
            return [[]]
        except:
            print("Unknown error with {}".format(self.path))
            return [[]]
        all_pages = []
        for g, page in page_group:
            words = self.page_to_list(page.reset_index())
            if max_tokens:
                words = words[:max_tokens]
            all_pages.append(words)
        return all_pages
        
    def page_to_list(self, page_df):
        all_word_list = [word for ind, row in page_df.iterrows() for word in ([row.lowercase]*row['count'])]
        random.shuffle(all_word_list)
        return all_word_list
    
class HathiCorpus(MOTESCorpus):
    
    DocClass = HathiBook