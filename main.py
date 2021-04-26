import tensorflow as tf
from model.core import Net
from network.rnn import TypoClassifier


core = TypoClassifier("./data")
net = Net("typo", "./", core)

net.train(100)
