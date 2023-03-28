from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = "环境/政策领域文本词法分析工具，支持词性识别/可视化/依存分析等功能"


setup(
    name="envpos",
    version="0.6",
    author="bihuaibin",
    author_email="bi.huaibin@foxmail.com",
    description="环境/政策领域文本词性标注、术语、专有名词识别工具",
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
    packages=["envpos"]
)

