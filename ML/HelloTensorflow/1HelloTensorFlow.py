import tensorflow as tf

def test(param):
    temp = 10
    print ("this is test func")
    print(param)

print("Hello Tensorflow");
hello = tf.constant("hello world")
print(hello)
print(hello.numpy())