import json
import numpy as np
import tensorflow as tf

from anomalydetection.DAE import DAE
from anomalydetection.utils.utils import fnorm
class RDAE:
    """
        Parameters
        ----------
        history:
            list of history objects given by the autoencoder while fitting
        lambda_:
            penalty for the sparsity error
        encoder_layer_size:
            list of layer sizes for the encoder
        decoder_layer_size:
            list of layer sizes for the decoder
        activation_function:
            activation function used for each layer
        regularizer:
            activity_regularizer for layers
        regularizer_penalty:
            penalty for the regularizer
        dropout:
            percentage of neurons dropped
        shrink:
            shrink function applied for S (l1shrink or l21shrink)
        verbose:
            boolean for verbose printing
    """
    def __init__(self, encoder_layer_size=None, decoder_layer_size=None, lambda_=None, error=1.0e-7,
                 activation_function='linear', regularizer='', regularizer_penalty=0.01, dropout=0, shrink='l1',
                 verbose=False,threshold=None, contamination=None, mu_=None):

        self.mu_ = mu_
        self.history = []
        self.lambda_ = lambda_
        self.encoder_neurons = encoder_layer_size
        self.decoder_neurons = decoder_layer_size
        self.activation_function = activation_function
        self.regularizer = regularizer
        self.regularizer_penalty = regularizer_penalty
        self.dropout = dropout
        self.error = error
        self.threshold = threshold
        self.contamination = contamination
        self.AE = DAE(encoder_layer_size=self.encoder_neurons, decoder_layer_size=self.decoder_neurons,
                      activation_function=self.activation_function, regularizer=self.regularizer, regularizer_penalty=self.regularizer_penalty,
                      dropout=self.dropout, threshold=self.threshold, contamination=self.contamination)
        if shrink == 'l1':
            from anomalydetection.utils.utils import l1shrink as shrink_function
        elif shrink == 'l21':
            from anomalydetection.utils.utils import l21shrink as shrink_function
        elif shrink == 'l21T':
            from anomalydetection.utils.utils import l21shrink
            shrink_function = lambda x, eps: l21shrink(x.T, eps).T
        else:
            from .utils.utils import shrink as shrink_function
        self.shrink_function = shrink_function
        self.verbose = verbose

    def compile(self, **kwargs):
        self.AE.compile(**kwargs)

    def fit_transform(self, x, inner_iteration=50, outer_iteration=10, batch_size=50):
        # check if x shape matches the input of the autoencoder
        assert x.shape[1] == self.encoder_neurons[0]
        assert x.ndim == 2

        self.STATS = {
            'primal': [],
            'dual': [],
        }

        self.L = np.zeros(x.shape)
        self.S = np.zeros(x.shape)

        # calculate mu
        if self.lambda_ is None:
            self.lambda_ = 1 / np.sqrt(max(x.shape))
        if self.mu_ is None:
            self.mu_ = x.size / (4.0 * np.linalg.norm(x, 1))

        if self.verbose:
            print(f"Lambda: {self.lambda_}; mu: {self.mu_}" )
        LS0 = self.L + self.S

        xfnorm = fnorm(x)
        for i in range(outer_iteration):
            if self.verbose:
                print(f"Iteration: {i}")
            self.L = x - self.S

            # Train the autoencoder with L
            self.history.append(self.AE.fit(x=self.L, y=self.L,
                                       epochs=inner_iteration,
                                       batch_size=batch_size,
                                       verbose=self.verbose,
                                       shuffle=True))
            # get L optimized
            self.L = self.AE.predict(self.L, verbose=self.verbose)

            # shrink S
            self.S = self.shrink_function(x=(x - self.L), eps=self.lambda_ / self.mu_)

            # criterion 1: Check if L and S are close enough to X
            primal = fnorm(x - self.L - self.S) / xfnorm
            # criterion 2: Check if L and S have converged since last interation
            dual = np.min([self.mu_, np.sqrt(self.mu_)]) * fnorm(LS0 - self.L - self.S) / xfnorm

            if self.verbose:
                print(f"Errors: 1. {primal}  2. {dual}")

            self.STATS['primal'].append(primal)
            self.STATS['dual'].append(dual)

            if primal < self.error and dual < self.error:
                break

            # save L + S for c2 next iter
            LS0 = self.L + self.S

        if self.verbose:
            if i < self.max_iter - 1:
                print(f'Converged in {i} steps')
            else:
                print('Reached maximum iterations')

        if self.contamination is not None or self.threshold is not None:
            self.decision_scores_, self.labels_ = self.AE.predict_scores(self.L)

        return self.L, self.S

    def predict(self, x):
        return self.AE.predict(x)

    def save_stats(self, path):
        open(path).write(json.dumps(self.STATS))

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from anomalydetection.utils.noise import add_noise
    from anomalydetection.utils.ad import get_stats
    import gc
    ot = 20
    it = 50
    bs = 150
    lr = 0.001
    drop = 0


    x = np.load('../data/mnist/data.npk', allow_pickle=True)
    y = np.load('../data/mnist/y.npk', allow_pickle=True)
    anomalies = (y != 4).astype(int)

    CONTAMINATION = y[y != 4].size / y.size

    lambdas = [0.001, 0.003, 0.005, 0.01, 0.05, 0.1, 1, 10, 30, 50, 70]
    corruptions = [0, 50, 100, 150, 200, 250]

    all_scores = []
    for c in corruptions:
        seed = np.random.randint(1000)
        corrupted_x = add_noise(x, c, seed)

        scores = []
        for l in lambdas:
            model = RDAE(encoder_layer_size=[784, 392, 196], decoder_layer_size=[392, 784], shrink='l1',
                activation_function='sigmoid', regularizer='', dropout=drop, verbose=False, lambda_=l)
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr))
            L, S = model.fit_transform(corrupted_x, batch_size=bs, inner_iteration=it, outer_iteration=ot)

            stats = get_stats(S,anomalies, CONTAMINATION)

            del model
            tf.keras.backend.clear_session()
            gc.collect()

            scores.append(stats['fscore'])
            print(f"Lambda: {l} Corruption: {c} fscore: {stats['fscore']}")
        all_scores.append(scores)

    print(all_scores)
    final_scores = np.array(all_scores)
    with open('rdae_lr_last_try.npy', 'wb') as f:
        np.save(f, final_scores)
