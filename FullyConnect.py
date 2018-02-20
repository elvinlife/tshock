"""    
tshock:FulllabelConnect
author:@elvinlife
time:since 2018/2/10
"""
import tshock as ts
import tensorflow as tf
import numpy as np

class Model(ts.AbsModel):
    def __init__(self,name, sess):
        super(Model,self).__init__(name, sess)
    def _setup(self):
        image = tf.placeholder(tf.float32, shape=(None,28*28))
        label = tf.placeholder(tf.int32, shape=(None, ))
        lin1 = ts.Linear(name='lin1', input_size=28*28, output_size=200)
        o1 = tf.nn.relu(lin1.setup(image))
        lin2 = ts.Linear(name='lin2', input_size=200, output_size=10)
        o2 = tf.nn.relu(lin2.setup(o1))
        proba = tf.nn.softmax(o2, dim=1)
        pred = tf.argmax(proba, axis=1, output_type=tf.int32)
        loss = ts.categorical_cross_entropy_loss(tf.one_hot(label, 10), proba)
        acc = tf.reduce_mean(tf.cast(tf.equal(label, pred), tf.float32))
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
        self._add_slot(
            'train',
            inputs = (image, label),
            outputs = (loss, acc),
            updates = optimizer
        )
        self._add_slot(
            'test',
            inputs = (image, label),
            outputs = (loss, acc)
        )
        self._sess.run(tf.global_variables_initializer())

def main():
    train_src = ts.NpzDataSrc('/Users/cyz/dataset/mnist/train.npz')
    test_src = ts.NpzDataSrc('/Users/cyz/dataset/mnist/test.npz')
    train_stream = ts.FileDataStream(train_src.load('image', 'label'))
    test_stream = ts.FileDataStream(test_src.load('image', 'label'))
    batch_size = 100
    with tf.Session() as sess:
        model = Model('mnist', sess)
        train = model.get_slot('train')
        test = model.get_slot('test')
        for epoch in range(1,20,1):
            print('epoch{}'.format(epoch))
            for i in range(train_stream.size // batch_size):
                train_img, train_lbl = train_stream.next_batch(batch_size)
                train_loss, train_acc = train.run(np.reshape(train_img, (-1,28*28)), train_lbl)
                if i % 1000:
                    test_img, test_lbl = test_stream.next_batch(batch_size)
                    test_loss, test_acc = test.run(np.reshape(test_img, (-1,28*28)), test_lbl)
                    print("test acc:{}, train acc:{}".format(test_acc, train_acc), end='\r')

if __name__ == '__main__':
    main()