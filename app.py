import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import random

images = tf.placeholder(tf.float32, [None, 62, 250, 1])
labels = tf.placeholder(tf.int32, [None, 5, 32])


# 第一个字符
def loss(start_p, end_p, no):
    conv1_1 = tf.layers.conv2d(images[:, :, start_p:end_p, :], 32, kernel_size=(3, 3), activation=tf.nn.relu)
    conv1_2 = tf.layers.conv2d(conv1_1, 64, kernel_size=(3, 3), activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1_2, (3, 3), (2, 2))

    dropout1 = tf.layers.dropout(pool1, rate=0.5)

    conv2_1 = tf.layers.conv2d(dropout1, 128, kernel_size=(3, 3), activation=tf.nn.relu)
    conv2_2 = tf.layers.conv2d(conv2_1, 256, kernel_size=(3, 3), activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2_2, (3, 3), (2, 2))


    # flatten1=tf.reshape(pool2,(None,-1))
    flatten1 = tf.contrib.layers.flatten(pool2)

    dense1 = tf.layers.dense(flatten1, 32)

    score = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels[:, no, :], logits=dense1))
    acc = tf.metrics.accuracy(labels=tf.argmax(labels[:, no, :], axis=1), predictions=tf.argmax(dense1, axis=1))

    return score, acc


score0, acc0 = loss(0, 80, 0)
score1, acc1 = loss(60, 110, 1)
score2, acc2 = loss(90, 175, 2)
score3, acc3 = loss(125, 205, 3)
score4, acc4 = loss(170, 250, 4)

score = tf.reduce_sum([score0, score1, score2, score3, score4])

accuracy = tf.reduce_mean([acc0, acc1, acc2, acc3, acc4])
# accuracy=tf.multiply(acc,0.2)

def get_file():
    files = os.listdir('./data/split/')
    fileId = random.randint(0, len(files))

    data = np.load('./data/split/data10000-page{}.npz'.format(fileId))
    n_split = int(0.9 * data['X'].shape[0])

    return data['X'][:n_split], data['y'][:n_split], data['X'][n_split:], data['y'][n_split:]

X_train, y_train, X_test, y_test=get_file()

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(score)


model_name = '5div'

saver = tf.train.Saver()

plt.ion()
plt.show()
x_draw = []
y_draw = []

with tf.Session() as sess:
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    if os.path.exists("./model/{}/{}.index".format(model_name, model_name)):
        saver.restore(sess, "./model/{}/{}".format(model_name, model_name))

    test_idx = 0
    test_batch_size = 50
    test_total = X_test.shape[0]
    test_pageNo = int(test_total / test_batch_size)

    batch_size = 50
    total = X_train.shape[0]
    pageNo = int(total / batch_size)
    print("{}{}{}".format(batch_size, total, pageNo))
    for i in range(100000):

        if i%200==199:
            X_train, y_train, X_test, y_test = get_file()

        start = (i % pageNo * batch_size) % total
        end = (i % pageNo * batch_size + batch_size) % total
        if end == 0:
            end = total

        startTime = time.time()
        _, score_val, accuracy_val = sess.run(
            [train_op, score, accuracy]
            , feed_dict={images: X_train[start:end], labels: y_train[start:end]}
        )

        print("no={}      ,loss={:.4f},accuracy={:.4f},time={:.2f}".format(i, score_val, accuracy_val,
                                                                           time.time() - startTime))
        if i % 50 == 0 and i > 0:
            test_start = (test_idx % test_pageNo * test_batch_size) % test_total
            test_end = (test_idx % test_pageNo * test_batch_size + test_batch_size) % test_total
            if test_end == 0:
                test_end = test_total
            test_idx = test_idx + 1

            score_val, accuracy_val = sess.run(
                [score, accuracy], feed_dict={images: X_test[test_start:test_end], labels: y_test[test_start:test_end]}
            )
            saver.save(sess, "./model/{}/{}".format(model_name, model_name))
            print(i, '       score_val={:.5f}'.format(score_val)
                  , '        accuracy_val=', accuracy_val
                  )
            x_draw.append(test_idx)
            y_draw.append(accuracy_val)

            plt.title('accuracy')
            plt.plot(x_draw, y_draw, color='b')

            plt.pause(0.1)
