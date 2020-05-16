import tensorflow as tf

with tf.device("/gpu:0"):
    import keras
with tf.Session() as sess:
    # Run your code