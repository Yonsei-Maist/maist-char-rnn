from setuptools import setup, find_packages

setup(
    name             = 'maist-char-rnn',
    version          = '1.0',
    description      = 'Char-RNN module for MAIST',
    author           = 'Chanwoo Gwon',
    author_email     = 'arknell@yonsei.ac.kr',
    url              = 'https://github.com/Yonsei-Maist/maist-char-rnn.git',
    install_requires = [
    ],
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['char', 'rnn'],
    python_requires  = '>=3',
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3.7'
    ]
)