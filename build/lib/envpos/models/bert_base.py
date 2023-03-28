from .model_base import ModelBase
from transformers import TrainingArguments, Trainer ,BertTokenizerFast, BertConfig
import torch
import os 
from tqdm import tqdm
import numpy as np #for np.concatnate
import random

class BertBase(ModelBase):
    def __init__(self,path,config = None,**kwargs):
        super().__init__()
        self.initialize_config(path)
        self.update_config(kwargs)
        self.update_config(config)
        self.initialize_tokenizer(path)
        self.initialize_bert(path)
        if 'max_length' not in kwargs:
            self.set_attribute(max_length = 512)
            
        if 'key_metric' not in kwargs:
            self.set_attribute(key_metric = 'loss')
            
    def initialize_tokenizer(self,path):
        '''
        初始化tokenizer
        '''
        self.tokenizer = BertTokenizerFast.from_pretrained(path,verbose=False) 



    def initialize_config(self,path):
        '''
        初始化config
        '''
        if os.path.exists(os.path.join(path,'config.json')):
            config = BertConfig.from_pretrained(path,verbose=False)
        else:
            config = BertConfig()
        config.update(self.config.to_diff_dict())
        self.config = config
    
    def align_config(self):
        '''
        对齐config，在initialize_bert的时候调用，如有必要则进行重写。
        '''
        pass
    
    def initialize_bert(self,path = None,config = None,**kwargs):
        '''
        初始化bert,需要继承之后重新实现
        Args:
            BertPretrainedModel `transformers.models.bert.modeling_bert.BertPreTrainedModel`:
                Hugging face transformer版本的 Bert模型
                默认为 BertForMaskedLM
                目前只支持pytorch版本
           path `str`:
               模型的路径，默认为None。如果不是None，优先从导入
           config `dict`:
               模型的配置
        '''
        if path is not None:
            self.update_model_path(path)
        
        if config is not None:
            self.update_config(config)
        
        self.update_config(kwargs)
        
        self.align_config()
            
        #例如：
        #self.model = BertPretrainedModel.from_pretrained(self.model_path)
 
        
    def add_spetial_tokens(self,tokens):
        self.tokenizer.add_special_tokens({'additional_special_tokens':tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

    
    def update_model_path(self,path):
        '''
        更新模型路径
       Args:
           path `str`:
               模型新的路径
        '''
        if path is not None:
            self.config._name_or_path = path
            
    @property
    def model_path(self):
        '''
        获得模型路径，没有路径则返回None
        '''
        if hasattr(self.config,'_name_or_path'):
            return self.config._name_or_path
        else:
            return None
    
    @property
    def label2id(self):
        '''
        返回一个dict,标签转id
        '''
        if hasattr(self.config,'label2id'):
            return self.config.label2id
        else:
            return None
        
    @property
    def id2label(self):
        '''
        返回一个dict,id转标签
        '''
        if hasattr(self.config,'id2label'):
            return self.config.id2label
        else:
            return None
        
    @property
    def labels(self):
        '''
        返回一个list,所有标签
        '''
        if hasattr(self.config,'labels'):
            return self.config.labels
        else:
            return None
        
    @property
    def entities(self):
        '''
        返回一个list,所有实体
        '''
        if hasattr(self.config,'entities'):
            return self.config.entities   
        else:
            return None
        
    @property
    def num_labels(self):
        '''
        返回一个list,所有标签
        '''
        if hasattr(self.config,'num_labels'):
            return self.config.num_labels
        else:
            return None

    @property
    def num_entities(self):
        '''
        返回一个list,所有标签
        '''
        if hasattr(self.config,'num_entities'):
            return self.config.num_entities
        else:
            return None

    def get_train_reports(self):
        '''
        获得训练集metrics报告
        '''
        raise NotImplemented
    
    def get_valid_reports(self):
        '''
        获得验证集metrics报告
        '''
        raise NotImplemented
        
        
    def _tokenizer_for_inference(self,texts,des):
        bar = tqdm(texts)
        bar.set_description(des)
        tokens = []
        for text in bar:
            tokens.append(self.tokenizer.encode(text,max_length = self.max_length,add_special_tokens=True,padding='max_length',truncation=True))
        tokens = torch.tensor(tokens)
        return tokens

    @torch.no_grad()
    def _inference_per_step(self,dataloader):
        self.model.eval()
        bar = tqdm(dataloader)
        bar.set_description("正在 Inference ...")
        preds = []
        for X in bar:
            X = X[0].to(self.device)
            predict = self.model(X)[0]
            preds.append(predict.clone().detach().cpu().numpy())
            
        preds = np.concatenate(preds,axis = 0)
        return preds
    
    def inference(self, texts = None, batch_size = 2, save_result = True ,**kwargs):
        '''
        推理数据集，更快的速度，更小的cpu依赖，建议大规模文本推理时使用。
        与self.predict() 的区别是会将数据打包为batch，并使用gpu(如有)进行预测，最后再使用self.postprocess()进行后处理，保存结果至self.result
        
        texts (`List[str]`): 数据集
            格式为列表
        '''        
        texts = self._align_input_texts(texts)
        
        #模型
        self.model = self.model.to(self.device)
        
        #预处理
        texts = [self.preprocess(text) for text in texts]
        
        #准备数据集
        from torch.utils.data import TensorDataset,DataLoader
        tokens = self._tokenizer_for_inference(texts,des = "正在Tokenizing...")
        dataset = TensorDataset(tokens)
        dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = False,drop_last = False)
        
        #推理
        preds = self._inference_per_step(dataloader)
        
        #保存results获得
        results = []
        for text,pred in tqdm(zip(texts,preds),desc="正在后处理..."):
            results.append(self.postprocess(text,pred))
        
        if save_result:
            for text,result in zip(texts,results):
                self.result[text] = result
       
        return results


            
    def _tokenizer_for_training(self,dataset):
        res = self.tokenizer(dataset['text'],max_length = self.max_length,padding='max_length',truncation=True)
        return res
    