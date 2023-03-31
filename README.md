### 环境领域词法分析工具

## 使用方式 QuickStart

### 1. 安装
先安装torch和transformers,
torch需要根据gpu和cuda版本选择合适的版本，如果不熟悉这些，可以先尝试用cpu版本的。
```bash
pip install torch
pip install transformers
pip install envpos
```
transformers的依赖比较多，如果出现`import error`，大概率是transformers的依赖没有装上，pip install 装上即可

**或者**可以直接从项目中安装
```
git clone https://gitee.com/bihuaibin/envpos
cd envpos && python setup.py install && cd ..
```

### 2. 词法分析
- 进行推理：（默认是albert模型，因为体积小，推理速度快）
```python
import envpos
s = '''
（一）重点行业绿色升级工程。以钢铁、有色金属、建材、石化化工等行业为重点，推进节能改造和污染物深度治理。推广高效精馏系统、高温高压干熄焦、富氧强化熔炼等节能技术，鼓励将高炉—转炉长流程炼钢转型为电炉短流程炼钢。
'''
envpos.cut(s)
```

- 可视化
```python
envpos.viz(s)
```
![](fig/viz.png)

- 进行多句推理：
```python
s = [
    "（三）城镇绿色节能改造工程。全面推进城镇绿色规划、绿色建设、绿色运行管理，推动低碳城市、韧性城市、海绵城市、“无废城市”建设。",
    "（四）交通物流节能减排工程。推动绿色铁路、绿色公路、绿色港口、绿色航道、绿色机场建设，有序推进充换电、加注（气）、加氢、港口机场岸电等基础设施建设。",
    "（八）煤炭清洁高效利用工程。要立足以煤为主的基本国情，坚持先立后破，严格合理控制煤炭消费增长，抓好煤炭清洁高效利用，推进存量煤电机组节煤降耗改造、供热改造、灵活性改造“三改联动”，持续推动煤电机组超低排放改造。"
]
result = envpos.cut(s)
```

- 切换bert模型，Bert模型准确率比albert高5个点以上，建议在硬件支持的情况下，优先使用bert模型。
权重下载地址：  
链接：https://pan.baidu.com/s/1hYEJUM04UdHLKCnnAGIijQ     
提取码：gqyz     
```python
envpos.change_bert("bert模型权重所在文件夹")
```


### 3. 依赖ddparser进行句法分析
安装ddparser
```bash
pip install ddparser
```
依存关系分析:
```python
s = "要立足以煤为主的基本国情，坚持先立后破，严格合理控制煤炭消费增长"
envpos.dp_cut(s)
```

可视化：
```python
s = "要立足以煤为主的基本国情，坚持先立后破，严格合理控制煤炭消费增长"
envpos.dp_viz(s)
```

详细的使用方式见`使用示例.ipynb`

## 类别设置和标注规则

本工具对环境领域文本设计41类标注，其中词性标注15类，实体标注14类，术语标注12类。类别的设置见标注手册。
对于标注策略，本项目构建过程中参考了GB/T 20532-2006《信息处理用现代汉语词类标记规范》、北京大学《现代汉语语料库加工规范》、微软亚洲研究院《中文文本标注规范》、斯坦福CTB树库标注规范等，积累了详细且丰富（目前整理后超过30页）的标注手册，会逐渐更新在项目的wiki里：
- [gitee](https://gitee.com/bihuaibin/envpos/wikis/环境词法标注手册)
- [github](https://github.com/celtics1863/envpos/wikis/环境词法标注手册)


| 词性 | n        | vn     | r    | v    | d    | a      | p    | f      | q    | m    | conj | u    | xc       | w    | ord |
| ---- | -------- | ------ | ---- | ---- | ---- | ------ | ---- | ------ | ---- | ---- | ---- | ---- | -------- | ---- | ---- |
|      | 普通名词 | 动名词 | 代词 | 动词 | 副词 | 形容词 | 介词 | 方位词 | 量词 | 数词 | 连词 | 助词 | 其他虚词 | 标点 | 序数词 |
|      |          |        |      |      |      |        |      |        |      |      |      |      |          |      |       |

| 实体 | time | loc  | per  | com    | org  | gov  | doc      | event | pro                            | ins             | means           | meet   | code | c              |
| ---- | ---- | ---- | ---- | ------ | ---- | ---- | -------- | ----- | ------------------------------ | --------------- | --------------- | ------ | ---- | -------------- |
|      | 时间 | 地点 | 人名 | 公司名 | 组织 | 政府 | 文件名称 | 事件  | 工程/<br />项目/<br />环境设施 | 设备/<br />工具 | 方法/<br />工艺 | 会议名 | 编码 | 其他<br />专名 |
|      |      |      |      |        |      |      |          |       |                                |                 |                 |        |      |                |

| 术语 | med            | phe            | pol    | microbe | plant | animal | desease | hy   | group | act            | policy         | b    | env                |
| ---- | -------------- | -------------- | ------ | ------- | ----- | ------ | ------- | ---- | ----- | -------------- | -------------- | ---- | ------------------ |
|      | 环境<br />介质 | 环境<br />现象 | 污染物 | 微生物  | 植物  | 动物   | 疾病    | 行业 | 群体  | 政策<br />行动 | 政策<br />工具 | 属性 | 其他<br />环境术语 |
|      |                |                |        |         |       |        |         |      |       |                |                |      |                    |


gitee上有镜像项目：
[envpos](https://gitee.com/bihuaibin/envpos)


## 准确率报告
### 1. 总体准确率
使用$F_1$作为评价指标

| 模型 | $F_1$ | 
| ---- | ---- |
| envBert | 91.7 |
| envBert-large | 91.0 |
| envAlbert | 85.2 |
| Bert | 89.4 |
| Bert-large | 89.8 |

envBert-Large不如Bert，可能原因是1. envBert-large预训练的时间不如envBert长。2. 大模型容易过拟合，3. 数据集仍然相对比较小，虽然已经有百万字标注语料，但是相对于bert-large模型而言，还是不够。。。




## 一些例子

识别结果截图：

![](fig/recognize.png)



和LAC制作词云对比，envpos可以识别出长词，不会将长词切分为散串
![envpos结果](fig/compare.png)


与DDparser结合进行句法分析：
![](fig/ddparser.svg)


## liscence

Apache 2.0


## TODO List

TODO :
- [x] 上传项目
- [x] 打包至pypi库
- [x] 上传标注文档在wiki里
- [ ] 添加自定义词表功能
  - [ ] 针对普通词表中的词采取输出权重抑制策略
  - [ ] 针对实体/术语采取贪心识别策略
- [ ] 集成至[envtext](https://github.com/celtics1863/envText)中。

