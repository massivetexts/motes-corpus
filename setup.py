import os
from setuptools import setup

setup(name='MOTES Corpus Tools',
      packages=["motes_corpus"],
      version='0.0.1',
      description="Tools related to corpus development for the MOTES divergent thinking scoring test.",
      url="https://github.com/massivetexts/motes-corpus",
      author="Peter Organisciak",
      author_email="peter.organisciak@du.edu",
      license="MIT",
      classifiers=[
        'Intended Audience :: Education',
        "Natural Language :: English",
        'License :: OSI Approved :: MIT License',
        "Operating System :: Unix",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.1',
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic"
        ],
      install_requires=["numpy", "pandas"]
)
