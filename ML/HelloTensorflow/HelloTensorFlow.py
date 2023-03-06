import tensorflow as tf

def test():
    print ("this is test func")

print("Hello Tensorflow");
hello = tf.constant("hello world")
print(hello)
print(hello.numpy())