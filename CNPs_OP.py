import math
import numpy as np
import tensorflow as tf

from datetime import datetime
from print_log import PrintLog
from conditional_neural_processes_my import NeuralProcess


class CNPs_Optimization:

    def __init__(self, func, para):
        # "para" is the diction of parameters which needs to be optimized.
        # "para" --> {'x_1': (0,10), 'x_2': (-1, 1),..., 'x_n':(Lower_bound, Upper_bound)}

        self.f = func
        self.begin_time = datetime.now()
        self.timestamps_list = []
        self.target_list = []
        self.parameters_list = []
        self.pop_list = []
        self.keys = list(para.keys())
        self.bounds = np.array(list(para.values()), dtype=np.float)
        self.dim = len(self.keys)
        self.plog = PrintLog(self.keys)

    def initialization(self, pop_size):
        para_value = np.empty((pop_size, self.dim))
        self.plog.print_header(initialization=True)
        for col, (lower, upper) in enumerate(self.bounds):
            para_value[:, col] = np.random.RandomState().uniform(lower, upper, pop_size)
        target_value = np.empty((pop_size, 1))
        for i in range(pop_size):
            target_value[i] = self.f(para_value[i])
            # print(target_value[i])
            # print(para_value[i])
            self.plog.print_step(para_value[i], target_value[i][0])

        return para_value, target_value

    def model_build(self):
        dim_r = 4
        dim_h_hidden = 128
        dim_g_hidden = 128

        x_context = tf.placeholder(tf.float32, shape=(None, self.dim))
        y_context = tf.placeholder(tf.float32, shape=(None, 1))
        x_target = tf.placeholder(tf.float32, shape=(None, self.dim))
        y_target = tf.placeholder(tf.float32, shape=(None, 1))
        neural_process = NeuralProcess(x_context, y_context, x_target, y_target, dim_r, dim_h_hidden, dim_g_hidden)

        return neural_process

    def model_run(self, neural_process, train_op, sess, x_init, y_init, n_iter_cnp=5001):
        num_init = len(y_init)
        plot_freq = 1000
        for iter in range(n_iter_cnp):
            N_context = np.random.randint(1, num_init, 1)
            # create feed_dict containing context and target sets
            feed_dict = neural_process.helper_context_and_target(x_init, y_init, N_context)
            # optimisation step
            a = sess.run(train_op, feed_dict=feed_dict)
            if iter % plot_freq == 0:
                print(a[1])

    def maximize(self, num_iter=3, pop_size=10, uncertain_rate=0.2):
        pop_para, pop_target = self.initialization(pop_size)
        self.pop_list.append(pop_para)
        pop_max = np.max(pop_target)
        self.target_list.append(pop_max)
        idx_max = np.where(pop_target == pop_max)[0]
        max_para = np.squeeze(pop_para[idx_max])
        self.parameters_list.append(max_para.tolist())
        elapse_time = (datetime.now() - self.begin_time).total_seconds()
        self.timestamps_list.append(elapse_time)

        self.plog.print_header(initialization=False)

        cnp_model = self.model_build()
        sess = tf.Session()
        train_op_and_loss = cnp_model.init_NP(learning_rate=0.001)
        init = tf.global_variables_initializer()
        sess.run(init)

        n_op_iter = num_iter
        for iter_op in range(n_op_iter):
            self.model_run(cnp_model, train_op_and_loss, sess, x_init=pop_para, y_init=pop_target)

            num_candidate = 1000
            x_candidate = np.empty((num_candidate, self.dim))
            for col, (lower, upper) in enumerate(self.bounds):
                x_candidate[:, col] = np.random.RandomState().uniform(lower, upper, num_candidate)
            predict_candidate = cnp_model.posterior_predict(pop_para, pop_target, x_candidate)
            _, y_candidate_mu, y_candidate_sigma = sess.run(predict_candidate)

            num_uncertain = int(pop_size*uncertain_rate)
            if num_uncertain < 1:
                num_uncertain = 1
            num_select = pop_size-num_uncertain
            y_candidate_mu = np.squeeze(y_candidate_mu)
            y_candidate_sigma = np.squeeze(y_candidate_sigma)
            ind_mu = np.argpartition(y_candidate_mu, -num_select)[-num_select:]
            x_mu_select = x_candidate[ind_mu]
            ind_sigma = np.argpartition(y_candidate_sigma, -num_uncertain)[-num_uncertain:]
            x_sigma_select = x_candidate[ind_sigma]

            x_select = np.unique(np.concatenate((x_mu_select, x_sigma_select), axis=0), axis=0)
            _, idx_d = np.unique(np.concatenate((pop_para, x_select), axis=0), axis=0, return_index=True)
            # remove the same item
            idx_d = idx_d - pop_para.shape[0]
            idx_d = np.delete(idx_d, np.where(idx_d < 0))
            x_select = x_select[idx_d]
            n_selected = np.shape(x_select)[0]      # final number of candidate which are selected.
            y_select = np.empty((n_selected, 1))
            for i in range(n_selected):
                y_select[i] = self.f(x_select[i])
                self.plog.print_step(x_select[i], y_select[i][0])

            self.pop_list.append(x_select)
            pop_max = np.max(y_select)
            self.target_list.append(pop_max)
            idx_max = np.where(y_select == pop_max)[0]
            max_para = np.squeeze(x_select[idx_max])
            self.parameters_list.append(max_para.tolist())
            elapse_time = (datetime.now() - self.begin_time).total_seconds()
            self.timestamps_list.append(elapse_time)

            pop_para = np.concatenate((pop_para, x_select), axis=0)
            pop_target = np.concatenate((pop_target, y_select), axis=0)
            print("The %d-th iteration is completed!" %iter_op)



