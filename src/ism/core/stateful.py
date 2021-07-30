import numpy as np
from tensorflow.keras.callbacks import Callback


class StatefulMixin:
    def _initialize_states(self):
        self.initial_states = []
        self.n_states = len(self.model.layers[0].states)
        batch_size = self.pretrain_batch_size - self.sub_batch_size + 1

        for _ in range(self.n_states):
            self.initial_states.append([
                np.zeros(shape=(batch_size, layer.units), dtype=np.float32)
                for layer in self.model.layers
            ])

    def _update_state(self, model):
        batch_size = self.batch_size - self.sub_batch_size + 1

        for si in range(self.n_states):
            self.initial_states[si] = [
                np.repeat(model.layers[li].states[si].numpy()[-1:, :], repeats=batch_size, axis=0)
                for li in range(len(model.layers))
            ]


class ResetStateCallback(Callback):
    def __init__(self, initial_states):
        self.initial_states = initial_states

    def on_train_batch_begin(self, batch, logs=None):
        self._reset_states()

    def on_predict_batch_begin(self, batch, logs=None):
        self._reset_states()

    def _reset_states(self):
        for i in range(len(self.model.layers)):
            self.model.layers[i].reset_states([state[i] for state in self.initial_states])
