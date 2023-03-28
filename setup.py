from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

    
    
setup(
    name="envpos",
    version="0.51",
    author="bihuaibin",
    author_email="bi.huaibin@foxmail.com",
    description="政策/环境领域文本词性标注、术语、专有名词识别工具",
    long_description = long_description,
    # 项目主页
    url="https://github.com/celtics1863/envpos", 
    install_requires=[
        "huggingface_hub",
        "packaging",
        "palettable",
        "pytorch-crf",
        "numpy",
        "pandas",
        "tqdm"
    ],
    include_package_data=True,
    python_requires='>=3.6',
    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=["envpos"]
)

