from setuptools import setup, find_packages

setup(
    name             = 'char-rnn',
    version          = '1.0',
    description      = 'Char-RNN module',
    author           = 'Chanwoo Gwon',
    author_email     = 'arknell@yonsei.ac.kr',
    url              = 'https://github.com/Yonsei-Maist/char-rnn.git',
    install_requires = [
        "tensorflow>=2.3.1"
    ],
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['char', 'rnn'],
    python_requires  = '>=3',
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3.7'
    ]
)