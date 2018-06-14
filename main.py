import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

class Converter:
    n = None
    name2id = None
    id2name = None

    def __init__(self):
        self.n = 0
        self.name2id = dict()
        self.id2name = dict()

    def get_id(self, name, insert=False):
        if name not in self.name2id:
            if insert:
                id = self.n
                self.name2id[name] = id
                self.id2name[id] = name
                self.n += 1
            else:
                return None
        else:
            id = self.name2id[name]
        return id


if __name__ == '__main__':
    result = np.recfromcsv('./datasets/results.csv', delimiter=',', skip_header=1)
    print(len(result))

    team_conv = Converter()
    X_year = list()
    X_hid = list()
    X_aid = list()
    Y_score = list()
    for row in result:
        str_date = row[0].decode("utf-8")
        date = datetime.datetime.strptime(str_date, '%Y-%m-%d')
        home_name = row[1].decode("utf-8")
        away_name = row[2].decode("utf-8")
        home_score = row[3]
        away_score = row[4]

        home_id = team_conv.get_id(home_name, insert=True)
        away_id = team_conv.get_id(away_name, insert=True)
        X_year.append(date.year)
        X_hid.append(home_id)
        X_aid.append(away_id)
        Y_score.append(np.array([home_score, away_score]))

    import pprint
    pprint.pprint(team_conv.name2id)
    n_team = team_conv.n


    def fc(x, num_out):
        return layers.fully_connected(
            x, num_out,
            activation_fn=None
        )

    input_year = tf.placeholder(tf.float32, [None])
    input_hid = tf.placeholder(tf.int32, [None])
    input_aid = tf.placeholder(tf.int32, [None])
    input_score = tf.placeholder(tf.float32, [None, 2])

    oh_hid = tf.one_hot(input_hid, depth=n_team)
    oh_aid = tf.one_hot(input_aid, depth=n_team)

    z0 = tf.concat(
        (
            tf.expand_dims(input_year, axis=1),
            oh_hid,
            oh_aid
        ),
        axis=1
    )

    z1 = fc(z0, 64)
    z2 = fc(z1, 64)
    z3 = fc(z2, 32)
    output_score = fc(z3, 2)
    loss = tf.losses.huber_loss(
        labels=input_score,
        predictions=output_score
    )
    '''
    loss = tf.reduce_mean(
        tf.square(
            output_score - input_score
        )
    )
    '''

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_step = optimizer.minimize(loss)

    saver = tf.train.Saver()

    mdl_path = './fifa.mdl'
    with tf.Session() as sess:
        try:
            saver.restore(sess, mdl_path)
        except:
            sess.run(tf.global_variables_initializer())

        # split
        L = len(X_hid) // 2
        for epoch in range(0):
            _, trl = sess.run(
                [
                    train_step,
                    loss
                ],
                feed_dict={
                    #input_year: X_year[:L],
                    #input_hid: X_hid[:L],
                    #input_aid: X_aid[:L],
                    #input_score: Y_score[:L]
                    input_year: X_year,
                    input_hid: X_hid,
                    input_aid: X_aid,
                    input_score: Y_score
                }
            )
            tel = sess.run(
                loss,
                feed_dict={
                    input_year: X_year[L:],
                    input_hid: X_hid[L:],
                    input_aid: X_aid[L:],
                    input_score: Y_score[L:]
                }
            )
            print(epoch, trl, tel)

        saver.save(sess, mdl_path)

        score = sess.run(
            output_score,
            feed_dict={
                input_year: [2018, 2018, 2018, 2018, 2018, 2018],
                # benefits
                #input_hid: [22, 22, 22,     47, 6,    6],
                #input_aid: [161, 47, 6,     161, 161, 47]

                # no benefits
                input_hid: [22, 47, 6,     47, 6,    6],
                input_aid: [161, 22, 22,     161, 161, 47]
            }
        )
        print('score',  score)


    #data = np.recfromcsv('./datasets/World Cup 2018 Dataset.csv', delimiter=',', skip_header=21)

    data = list()
    with open('./datasets/World Cup 2018 Dataset.csv') as f:
        lcnt = 0
        for line in f.readlines():
            if lcnt > 20:
                row = line.split(',')
                data.append(row)
                #print(row)
            lcnt += 1


    for row in data:
        name = row[0]
        if name == 'Columbia':
            name = 'Colombia'
        if name == 'Costarica':
            name = 'Costa Rica'
        if name =='IRAN':
            name = 'Iran'
        if name == 'Korea':
            name = 'Korea Republic'
        if name == 'Porugal':
            name = 'Portugal'

        id = team_conv.get_id(name)
        #print(name, id)

