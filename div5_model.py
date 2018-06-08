from model import Model
import tensorflow as tf
import os, random, time, sys
import numpy as np
import matplotlib.pyplot as plt


class Div5Model(Model):
    def __init__(self, model_name, show_val):
        self.images = tf.placeholder(tf.float32, [None, 62, 250, 1])
        self.labels = tf.placeholder(tf.int32, [None, 5, 32])
        self.model_name = model_name
        self.show_val = show_val

        self.loss_vals = []
        self.accuracy_vals = []

        if not os.path.exists('./model/{}'.format(self.model_name)):
            os.makedirs('./model/{}'.format(self.model_name))

        if os.path.exists('./model/{}/{}.npz'.format(self.model_name, self.model_name)):
            vals=np.load('./model/{}/{}.npz'.format(self.model_name,self.model_name))
            self.loss_vals,self.accuracy_vals=vals['loss_vals'].tolist(),vals['accuracy_vals'].tolist()

        if self.show_val:
            plt.ion()
            plt.show()

        self._draw_line()

    def train(self):
        loss, accuracy = self.build()
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            saver = tf.train.Saver()

            if os.path.exists('./model/{}/{}.index'.format(self.model_name, self.model_name)):
                saver.restore(sess, './model/{}/{}'.format(self.model_name, self.model_name))

            x_train, y_train, x_test, y_test = self._get_file();

            test_idx = 0
            test_batch_size = 50
            test_total = x_test.shape[0]
            test_page_no = int(test_total / test_batch_size)

            batch_size = 50
            total = x_train.shape[0]
            page_no = int(total / batch_size)
            print("{}{}{}".format(batch_size, total, page_no))
            for i in range(sys.maxsize):
                if i % 200 == 199:
                    x_train, y_train, x_test, y_test = self._get_file()

                start = (i % page_no * batch_size) % total
                end = (i % page_no * batch_size + batch_size) % total
                if end == 0:
                    end = total

                start_time = time.time()
                _, loss_val, accuracy_val = sess.run(
                    [train_op, loss, accuracy]
                    , feed_dict={self.images: x_train[start:end], self.labels: y_train[start:end]}
                )

                print("no={}      ,loss={:.4f},accuracy={:.4f},time={:.2f}".format(i, loss_val, accuracy_val,
                                                                                   time.time() - start_time))
                if i % 20 == 0 and i > 0:
                    test_start = (test_idx % test_page_no * test_batch_size) % test_total
                    test_end = (test_idx % test_page_no * test_batch_size + test_batch_size) % test_total
                    if test_end == 0:
                        test_end = test_total
                    test_idx = test_idx + 1

                    loss_val, accuracy_val = sess.run(
                        [loss, accuracy],
                        feed_dict={self.images: x_test[test_start:test_end], self.labels: y_test[test_start:test_end]}
                    )

                    saver.save(sess, "./model/{}/{}".format(self.model_name, self.model_name))
                    print(i, '       loss_val={:.5f}'.format(loss_val)
                          , '        accuracy_val=', accuracy_val)

                    self.loss_vals.append(loss_val)
                    self.accuracy_vals.append(accuracy_val)

                    self._draw_line()

    def build(self):
        score0, acc0 = self._loss(0, 80, 0)
        score1, acc1 = self._loss(60, 110, 1)
        score2, acc2 = self._loss(90, 175, 2)
        score3, acc3 = self._loss(125, 205, 3)
        score4, acc4 = self._loss(170, 250, 4)

        score = tf.reduce_sum([score0, score1, score2, score3, score4])

        accuracy = tf.reduce_mean([acc0, acc1, acc2, acc3, acc4])

        return score, accuracy

    def _draw_line(self):
        if self.show_val:
            plt.subplot(2, 1, 1)
            plt.title('accuracy')
            plt.plot(range(len(self.accuracy_vals)), self.accuracy_vals, color='b')

            plt.subplot(2, 1, 2)
            plt.title('loss')
            plt.plot(range(len(self.loss_vals)), self.loss_vals, color='r')
            plt.pause(0.1)

        self._save_vals()

    def _save_vals(self):
        np.savez_compressed('./model/{}/{}.npz'.format(self.model_name, self.model_name),
                            loss_vals=np.array(self.loss_vals),accuracy_vals=np.array(self.accuracy_vals))

    def _loss(self, start_p, end_p, no):
        conv1_1 = tf.layers.conv2d(self.images[:, :, start_p:end_p, :], 32, kernel_size=(3, 3), activation=tf.nn.relu)
        conv1_2 = tf.layers.conv2d(conv1_1, 32, kernel_size=(3, 3), activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1_2, (3, 3), (2, 2))

        conv2_1 = tf.layers.conv2d(pool1, 128, kernel_size=(3, 3), activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(conv2_1, 128, kernel_size=(3, 3), activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2_2, (3, 3), (2, 2))

        # flatten1=tf.reshape(pool2,(None,-1))
        flatten1 = tf.contrib.layers.flatten(pool2)
        dense1 = tf.layers.dense(flatten1, 1024)
        dropout1 = tf.layers.dropout(dense1, rate=0.5)
        dense2 = tf.layers.dense(dropout1, 32)
        score = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels[:, no, :], logits=dense2))
        acc = tf.metrics.accuracy(labels=tf.argmax(self.labels[:, no, :], axis=1),
                                  predictions=tf.argmax(dense2, axis=1))

        return score, acc

    def _get_file(self):
        files = os.listdir('./data/split/')
        file_id = random.randint(0, len(files)-1)

        data = np.load('./data/split/{}'.format(files[file_id]))
        n_split = int(0.9 * data['X'].shape[0])

        return data['X'][:n_split], data['y'][:n_split], data['X'][n_split:], data['y'][n_split:]


model = Div5Model('div5', True)

model.train()
