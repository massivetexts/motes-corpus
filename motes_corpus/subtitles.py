import re
from motes_corpus.modeling import MOTESCorpus

class Subtitle(object):

    pre_tokenized = False
    
    def __init__(self, path):
        try:
            with open(path, encoding='utf-8') as f:
                txt = f.read()
        except:
            # clunky - try latin-1 if utf-8 failed
            with open(path, encoding='latin-1') as f:
                txt = f.read()
        self.subs = re.findall('(\d+)\n(.*?)\n(.*?)\n\s*\n', txt, flags=re.DOTALL)

    @property
    def text(self):
        # Basic extraction for now
        # Todo - strip HTML tags, [action] tags, option for removing first line or x lines
        just_txt = [txt for n, time, txt in self.subs]
        return "\n".join(just_txt)

class SubtitleCorpus(MOTESCorpus):
    
    DocClass = Subtitle