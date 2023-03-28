pos_albert = "./pretrained_models/pos-albert"
pos_bert = "./pretrained_models/pos-bert"
pos_bert_download_url = "https://huggingface.co/celtics1863/pos-bert"


import os

basedir = os.path.dirname(__file__)


class FileConfig:
    def __init__(self):
#         print(os.path.normpath(env_vocab))
        pass
        
    @property
    def pos_albert(self):
        return self.get_abs_path(pos_albert)
    
    @property
    def pos_bert(self):
        path = self.get_abs_path(pos_bert)
        if os.path.exists(path):
            return path
        else:
            return pos_bert_download_url

    @property
    def pos(self):
        return self.pos_albert


    def get_abs_path(self,relative_path):
        return os.path.normpath(os.path.join(basedir,relative_path))