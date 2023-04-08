__version__ = '0.0.1'
__license__ = 'Apache Software License'

from .models import *
from .visualizers import *
from .files import FileConfig
from typing import *
import warnings
warnings.filterwarnings("ignore", message=".*FutureWarning:.*")


LABEL_MAPPING = {
            "med":"环境介质",
            "phe":"环境现象",
            "pol":"污染物",
            "microbe":"微生物",
            "plant":"植物",
            "animal":"动物",
            "desease":"疾病",
            "hy":"行业",
            "group":"群体",
            "act":"政策行动",
            "policy":"政策要求",
            "b":"抽象属性",
            "env":"其他专业术语",
            "time":"时间",
            "loc":"地点",
            "per":"人名",
            "com":"公司名",
            "org":"组织名",
            "gov":"政府部门",
            "doc":"文档名",
            "event":"事件名",
            "pro":"设施/项目名",
            "ins":"工具名",
            "means":"方法名",
            "meet":"会议名",
            "code":"编码",
            "n":"名词",
            "v":"动词",
            "a":"形容词",
            "d":"副词",
            "vn":"动名词",
            "f":"方位词",
            "p":"介词",
            "r":"代词",
            "m":"数词",
            "q":"量词",
            "conj":"连词",
            "w":"标点",
            "u":"助词",
            "xc":"虚词",
        }


class POS:
    def __init__(self):
        self.file_config = FileConfig()
        self.model = AlbertNER(self.file_config.pos_albert)
        self.ddp_vizer = DependencyVisualizer()

    def change_bert(self,path = None):
        if path is None:
            path = "https://huggingface.co/celtics1863/pos-bert"
        
        self.model = BertNER(path)
        
    def use_cuda(self):
        self.model.set_device("cuda")

    def cut(self,text:Union[str, List[str]],auto_group=True, **kwargs):
        '''
        str 或者 List[str]
            对str，返回Tuple(words, pos)
            对List[str]，返回[List[Tuple(words, pos )]
        '''
        result = self.model(text, print_result=False,auto_group=auto_group,**kwargs)
        if isinstance(text, str):
            return result[0]
        else:
            return result

    def viz(self,text : str):
        return self.model(text, print_result=True, return_result= False)


    def generate_html(self,s : str):
        '''
        可视化生成html
        '''
        words,poses = self.cut(s)
        return self.model.visualizer.generate_html(words, poses)

    def _init_ddp(self):
        if hasattr(self,"ddp"):
            return True
        else:
            try:
                from ddparser import DDParser
            except:
                assert 0,"请安装ddparser，使用命令：pip install ddparser"
            self.ddp = DDParser(use_pos=True)
    
    def dp_viz(self,s:str,save_path = "ddp.html"):
        '''
        可视化depencency visualizer
        '''
        self._init_ddp()
        
        words,poses = self.cut(s)
        
        res = self.ddp.parse_seg([words])[0]
        
        words = [{"word":w,"text":w, "tag":p}  for w,p in zip (words, poses)]

        words = [{"word":"HEAD","text":"HEAD","tag":"HEAD"}] + words

        id2words = {idx:word["word"] for idx,word in enumerate(words)}

        arcs = [
            {
                "start" : min(idx + 1,head),
                "end": max(idx + 1,head),
                "dir" : True if idx < head else False,
                "label": label
            }
            for idx,(head,label) in enumerate(zip(res["head"],res["deprel"]))
        ]
        
        self.ddp_vizer.render(0,words,arcs,save_path=save_path)

    
    def dp_cut(self,s):
        '''
        s: str or List(str)
        '''
        self._init_ddp()
        
        result = self.cut(s)
        if isinstance(s,str):
            result = [result]
            
        words = [w for w,p in result]
        poses = [p for w,p in result]
        
        ddp_result = self.ddp.parse_seg(words)
        
        for i in range(len(ddp_result)):
            ddp_result[i]["pos"] = poses[i]
        
        return ddp_result
    
    @property
    def labels(self):
        return LABEL_MAPPING
        
        
pos = POS()
cut = pos.cut
generate_html = pos.generate_html
change_bert = pos.change_bert
use_cuda = pos.use_cuda
viz = pos.viz
dp_cut = pos.dp_cut
dp_viz = pos.dp_viz