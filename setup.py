from setuptools import setup, find_packages

setup(
    name="envpos",
    version="0.5",
    author="bihuaibin",
    author_email="bi.huaibin@foxmail.com",
    description="政策/环境领域文本词性标注、术语、专有名词识别工具",
    # 项目主页
    url="https://github.com/celtics1863/envpos", 
    install_requires=[
        'pytorch >= 1.8',
        "transformers >= 3.0",
        "datasets",
        "huggingface_hub",
        "packaging",
        "palettable",
        "pytorch-crf",
        "numpy",
        "pandas",
    
    ],

    include_package_data=True,
    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)

