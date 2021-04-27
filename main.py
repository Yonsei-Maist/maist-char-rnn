import tensorflow as tf
from model.core import Net
from network.rnn import TypoClassifier


core = TypoClassifier("./data")
net = Net("typo", "./", core)

# net.train(200)

# res = net.test(199)

vec = core.to_vector('바나ㄴㅏ', True)
print(vec)
res = net.predict(199, tf.convert_to_tensor([vec], dtype=tf.int64))
print(core.get_label(tf.math.argmax(res, axis=1).numpy()))
