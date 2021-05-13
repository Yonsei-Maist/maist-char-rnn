import tensorflow as tf
from model.core import Net
from network.rnn import TypoClassifier


core = TypoClassifier("./data")
net = Net("typo", "./", core)

# net.train(150)

# res = net.test(149)

# print(res)
vec = core.to_predictable_data('ㅍㄷ')
print(vec)
res = net.predict(149, vec)
print(core.get_label(tf.math.argmax(res, axis=1).numpy()))
