import setuptools
from tlidb.version import __version__

print(f"Version {__version__}")


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="tlidb",
    version=__version__,
    license='MIT',
    url='https://github.com/alon-albalak/TLiDB',
    author="Alon Albalak",
    author_email="alon_albalak@ucsb.edu",
    description="The Transfer Learning in Dialogue Baselines Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [
        'nltk==3.6.5',
        'scikit-learn==1.0',
        'transformers==4.11.3',
        'torch>=1.10',
        'sentencepiece==0.1.96',
        'bert-score==0.3.11'
    ],
    packages=setuptools.find_packages(exclude=['dataset_preprocessing']),
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['tlidb=tlidb.examples.__main__:main']
    },
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
