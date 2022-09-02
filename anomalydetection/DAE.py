import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
import pandas as pd
import numpy as np

class DAE(tf.keras.Model):
    def __init__(self,  encoder_layer_size=None, decoder_layer_size=None,
                 activation_function='tanh',
                 regularizer='', regularizer_penalty=0.01,
                 dropout=None, threshold=None, contamination=None):
        super().__init__()

        self.labels_ = None
        self.decision_scores_ = None
        self.encoder_layer_sizes = encoder_layer_size
        self.decoder_layer_sizes = decoder_layer_size
        self.activation_function = activation_function
        self.regularizer = regularizer
        self.regularizer_penalty = regularizer_penalty
        self.dropout = dropout

        self.threshold = threshold
        self.contamination = contamination

        #self._check_parameters()
        self._create_model()

    def _check_parameters(self):
        if not isinstance(self.encoder_layer_sizes, list):
            raise TypeError(f"Encoder_layer_sizes expected a list of int; recieved {self.encoder_layer_sizes}")

        if not isinstance(self.decoder_layer_sizes, list):
            raise TypeError(f"Decoder_layer_sizes expected a list of int; recieved {self.decoder_layer_sizes}")

        if not isinstance(self.regularizer, str) or self.regularizer not in ['l1', 'l2']:
            raise TypeError("Regularizer should be a string of value: \"l1\" or \"l2\" ")

        if not isinstance(self.regularizer_penalty, int):
            raise TypeError("Penalty should be an integer")

        if not isinstance(self.dropout, float) or not (0.0 < self.dropout < 1.0):
            raise TypeError("Dropout should be a float between 0 and 1")

        if not isinstance(self.threshold, int):
            raise TypeError("Threshold should be an integer")

        if not isinstance(self.contamination, int) or not (0 < self.contamination < 1):
            raise TypeError("Contamination should be an integer")

    def _create_model(self):
        if self.regularizer == "l1":
            from tensorflow.keras.regularizers import L1
            self.activity_regularizer = L1(self.regularizer_penalty)
        elif self.regularizer == "l2":
            from tensorflow.keras.regularizers import L2
            self.activity_regularizer = L2(self.regularizer_penalty)
        else:
            self.activity_regularizer = None

        self.encoder = tf.keras.Sequential([
            Input(shape=self.encoder_layer_sizes[0])
        ])
        for shape in self.encoder_layer_sizes[1:]:
            if self.activity_regularizer:
                self.encoder.add(Dense(units=shape,
                                       activation=self.activation_function,
                                       activity_regularizer=self.activity_regularizer))
            else:
                self.encoder.add(Dense(units=shape,
                                       activation=self.activation_function))

            if self.dropout is not None and self.dropout > 0.0:
                self.encoder.add(Dropout(self.dropout))

        # decoder layer
        self.decoder = tf.keras.Sequential()
        for shape in self.decoder_layer_sizes:
            if self.activity_regularizer:
                self.decoder.add(Dense(units=shape,
                                       activation=self.activation_function,
                                       activity_regularizer=self.activity_regularizer))
            else:
                self.decoder.add(Dense(units=shape,
                                       activation=self.activation_function))

            if self.dropout is not None and self.dropout > 0.0:
                self.decoder.add(Dropout(self.dropout))

    def call(self, inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)

    def summary(self):
        print(self.encoder.summary())
        print(self.decoder.summary())

    def predict_scores(self, x):
        x_prime = super().predict(x)
        self.decision_scores_ = np.linalg.norm(x - x_prime, ord=2, axis=1)

        if self.threshold is None:
            if self.contamination is not None:
                self.threshold = pd.Series(self.decision_scores_).quantile(1 - self.contamination)

        if self.threshold is not None:
            self.labels_ = (self.decision_scores_ > self.threshold).astype(int)
        else:
            return

        return self.decision_scores_, self.labels_
