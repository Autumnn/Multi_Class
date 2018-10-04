import tensorflow as tf
import numpy as np


class NeuralProcess:

    def __init__(self, x_context, y_context, x_target, y_target,
                 dim_r, dim_h_hidden, dim_g_hidden):

        self.dim_r = dim_r
        self.dim_h_hidden = dim_h_hidden
        self.dim_g_hidden = dim_g_hidden
        self.x_context = x_context
        self.y_context = y_context
        self.x_target = x_target
        self.y_target = y_target
        self.x_all = tf.concat([self.x_context, self.x_target], axis=0)
        self.y_all = tf.concat([self.y_context, self.y_target], axis=0)

    def map_xy_to_z_params(self, x, y):
        inp = tf.concat([x, y], axis=1)
        t_x_1 = tf.layers.dense(inp, self.dim_h_hidden, tf.nn.sigmoid, name='encoder_layer_1', reuse=tf.AUTO_REUSE)
        t_x = tf.layers.dense(t_x_1, self.dim_r, name="encoder_layer_2", reuse=tf.AUTO_REUSE)
        r_1 = tf.reduce_mean(t_x, axis=0)
        r = tf.reshape(r_1, shape=(1, -1))   ######################

        #size = tf.shape(t_x)

        #return {'mu': mu, 'sigma': sigma, 'size': size}
        return r


    def g(self, r_sample, x_star, noise_sd = 0.05):
        # inputs dimensions
        # r_sample has dim [1, dim_z]
        # x_star has dim [N_star, dim_x]

        N_star = tf.shape(x_star)[0]

        # z_sample_rep will have dim [N_star, dim_z]
        #r_sample_rep = tf.tile(tf.expand_dims(r_sample, axis=0), [N_star, 1])
        r_sample_rep = tf.tile(r_sample, [N_star, 1])
        # x_star_rep will have dim [N_star, dim_x]

        inp = tf.concat([x_star, r_sample_rep], axis=1)

        hidden = tf.layers.dense(inp, self.dim_g_hidden, tf.nn.sigmoid, name="decoder_layer_1", reuse=tf.AUTO_REUSE)
       # hidden = tf.layers.dense(hidden_1, self.dim_g_hidden, tf.nn.sigmoid, name="decoder_layer_2", reuse=tf.AUTO_REUSE)

        mu_star = tf.layers.dense(hidden, 1, name="decoder_layer_mu", reuse=tf.AUTO_REUSE) # dim = [N_star, 1]
        sigma_star = tf.layers.dense(hidden, 1, tf.nn.sigmoid, name="decoder_layer_sigma", reuse=tf.AUTO_REUSE) # dim = [N_star, 1]
        #size = tf.shape(mu_star)
        #y_star = tf.squeeze(y_star, axis=2)
        #y_star = tf.transpose(y_star)         # dim = [N_star, 2]

        return {'mu': mu_star, 'sigma': sigma_star}
        #return {'y_star': y_star}

    '''
    def klqp_gaussian(self, mu_q, sigma_q, mu_p, sigma_p):
        sigma2_q = tf.square(sigma_q) + 1e-16
        sigma2_p = tf.square(sigma_p) + 1e-16
        temp = sigma2_q / sigma2_p + tf.square(mu_q - mu_p) / sigma2_p - 1.0 + tf.log(sigma2_p / sigma2_q + 1e-16)
        temp = 0.5 * tf.reduce_sum(temp)
        return temp
    '''

    def custom_objective(self, y_pred_params):
        mu = y_pred_params['mu']     # dim = [N_star, n_draws]
        sigma = y_pred_params['sigma']    # dim = N_star
        sdv = tf.sqrt(sigma)
        p_normal = tf.distributions.Normal(loc=mu, scale=sdv)
        p_star = p_normal.log_prob(self.y_all)
        loglik = tf.reduce_sum(p_star)
        loss = tf.negative(loglik)
        return loss


    def init_NP(self, learning_rate=0.001):
        r_context = self.map_xy_to_z_params(self.x_context, self.y_context)
        #r_all = self.map_xy_to_z_params(self.x_all, self.y_all)

        #epsilon = tf.random_normal(shape=(7, self.dim_z))
        #epsilon = tf.random_uniform(shape=(7, self.dim_z), minval=-10, maxval=10)


        #y_pred_params = self.g(z_sample, self.x_target)     # dim = [N_star, n_draws]
        y_pred_params = self.g(r_context, self.x_all)  # dim = [N_star, n_draws]

        loss = self.custom_objective(y_pred_params=y_pred_params)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer()

        train_op = optimizer.minimize(loss=loss)

        return [train_op, loss]


    def posterior_predict(self, x, y, x_star_value, epsilon=None, n_draws=1):
        # inputs for prediction time
        x_obs = tf.constant(x, dtype=tf.float32)
        y_obs = tf.constant(y, dtype=tf.float32)
        x_star = tf.constant(x_star_value, dtype=tf.float32)

        # for out-of-sample new points
        r_params = self.map_xy_to_z_params(x_obs, y_obs)
        #z_params_shape = tf.shape(z_params['mu'])

        # the source of randomness can be optionally passed as an argument
        if epsilon is None:
            #epsilon = tf.random_normal(shape=(n_draws, self.dim_r))
            epsilon = tf.random_normal(shape=(n_draws, 1))

        # predictions
        y_params = self.g(r_params, x_star)
        y_star = tf.add(y_params['mu'], tf.matmul(y_params['sigma'], tf.transpose(epsilon)))

        return y_star, y_params['mu'], y_params['sigma']
        #return y_star


    def helper_context_and_target(self, x, y, N_context):
        N = len(y)
        ori = np.linspace(1, N, N, dtype=int)
        context_set = np.random.choice(N, N_context, replace=False) + 1
        context_lef = np.setdiff1d(ori, context_set)

        dic = {self.x_context: x[context_set-1], self.y_context: y[context_set-1],
               self.x_target: x[context_lef-1], self.y_target: y[context_lef-1]}

        return dic






