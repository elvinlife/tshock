"""    
tshock:Lenet-5
author:@elvinlife
time:since 2018/2/26
"""
import tshock as ts
import tensorflow as tf

class Lenet(ts.AbsModel):
    def __init__(self,name, sess):
        super(Lenet,self).__init__(name, sess)
    def _setup(self):
        image = tf.placeholder(ts.float_default, shape=(None,28,28))
        label = tf.placeholder(ts.int_default, shape=(None, ))
        x = ts.image.to_dense(image)
        conv1 = ts.Conv2d((28, 28, 1), 6, filter_height=5, filter_width=5, name='conv1').setup(x)
        o1 = tf.nn.relu(ts.image.sub_sample(conv1, 2, 2))  #12*12*6
        conv2 = ts.Conv2d((12, 12, 6), 16, name='conv2').setup(o1)
        o2 = tf.nn.relu(ts.image.sub_sample(conv2, 2, 2))   #5*5*16
        conv3 = ts.Conv2d((5, 5, 16), 120, filter_height=5, filter_width=5, name='conv3').setup(o2)
        lin1 = ts.Linear(120, 10).setup(tf.reshape(conv3, (-1, 120)))
        proba = tf.nn.softmax(tf.tanh(lin1))
        pred = tf.argmax(proba, axis=1, output_type=ts.int_default)
        loss = ts.categorical_cross_entropy_loss(tf.one_hot(label, 10), proba)
        acc = tf.reduce_mean(tf.cast(tf.equal(label, pred), ts.float_default))
        optimizer = tf.train.AdamOptimizer(0.00001).minimize(loss)
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
        model = Lenet('mnist', sess)
        train = model.get_slot('train')
        test = model.get_slot('test')
        for epoch in range(1,20,1):
            print('epoch{}'.format(epoch), end='\n')
            for i in range(train_stream.size // batch_size):
                train_img, train_lbl = train_stream.next_batch(batch_size)
                train_loss, train_acc = train.run(train_img, train_lbl)
                if i % 100 == 0:
                    test_img, test_lbl = test_stream.next_batch(batch_size)
                    test_loss, test_acc = test.run(test_img, test_lbl)
                    print("test acc:%0.4f, train acc:%0.4f, test loss:%0.4f, train loss:%0.4f"
                          % (test_acc, train_acc, test_loss, train_loss), end='\r')

if __name__ == '__main__':
    main()