from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from ism.core.sub_batching import sub_batch, unroll_predictions
from ism.core.stateful import StatefulMixin, ResetStateCallback


class IncrementalRecurrentClassifier(ABC, BaseSKMObject, ClassifierMixin, StatefulMixin):
    """Incremental vanilla RNN classifier with scikit-flow API and based on Tensorflow/Keras API.

    This is an abstract class. In order to implement a child class, one have to set the
    `recurrrent_layer_class` property on it. 

    Attributes:
        n_classes: number of classes in the dataset to be used.
        n_features: number of features in the dataset to be used.
        recurrent_layers_spec: list of dictionaries layer setup, with `units` and `activation`.
        optimizer_class: instance of `tensorflow.keras.optimizers` to be used during optimization.
        optimizer_kwargs: arguments to be passed to `optimizer_class`.
        pretrain_n_epochs: number of epochs to run in the pre-training batch.
        n_epochs: number of epochs to run in the batches following the pre-train.
        pretrain_batch_size: size of the pre-train batch.
        batch_size: size of the remaining batches.
        sub_batch_size: size of the sub-batch, used by the sub-batching procedure of the ISM.
        verbosity: verbosity of the `fit` method.
        stateful: boolean flag indicating whether the statefulness will be activated.
    """
    def __init__(self, n_classes, n_features, recurrent_layers_spec, optimizer_class=None,
                 optimizer_kwargs=None, pretrain_n_epochs=1, n_epochs=1, pretrain_batch_size=1,
                 batch_size=1, sub_batch_size=1, verbosity=0, stateful=True):
        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.pretrain_n_epochs = pretrain_n_epochs
        self.n_epochs = n_epochs
        self.pretrain_batch_size = pretrain_batch_size
        self.batch_size = batch_size
        self.verbosity = verbosity
        self.sub_batch_size = sub_batch_size
        self.stateful = stateful

        if pretrain_batch_size > 0:
            self.first_run = True
        else:
            self.first_run = False

        self.pretrain_model = self._build_model(pretrain_batch_size, recurrent_layers_spec)
        self.model = self._build_model(batch_size, recurrent_layers_spec)
        self._compile_models(optimizer_class, optimizer_kwargs)

        if self.stateful:
            self._initialize_states()

    @property
    @abstractmethod
    def recurrrent_layer_class(self):
        """A sub-class of `tensorflow.keras.layers.RNN`."""
        raise NotImplementedError()

    @property
    def callbacks(self):
        if self.stateful:
            return [ResetStateCallback(self.initial_states)]
        return None

    def _build_model(self, batch_size, recurrent_layers_spec):
        recurrent_layers = self._create_recurrent_layers(batch_size, recurrent_layers_spec)
        output_layer = self._create_output_layer(self.n_classes)
        all_layers = recurrent_layers + [output_layer]
        return Sequential(all_layers)

    def _create_recurrent_layers(self, batch_size, recurrent_layers_spec):
        new_batch_size = batch_size - self.sub_batch_size + 1
        layers = []

        for i, layer_spec in enumerate(recurrent_layers_spec):
            layer_spec['stateful'] = True
            layer_spec['return_sequences'] = True
            layer_spec['dtype'] = 'float32'
            if i == 0:
                layer_spec['batch_input_shape'] = (new_batch_size, self.sub_batch_size, self.n_features)
            layers.append(self.recurrrent_layer_class(**layer_spec))

        return layers

    def _create_output_layer(self, n_classes):
        if n_classes > 1:
            return self.recurrrent_layer_class(
                stateful=True,
                return_sequences=True,
                dtype='float32',
                units=self.n_classes,
                activation='softmax'
            )
        return self.recurrrent_layer_class(
            stateful=True,
            return_sequences=True,
            dtype='float32',
            units=1,
            activation='sigmoid'
        )

    def _compile_models(self, optimizer_class, optimizer_kwargs):
        pretrain_loss_fn = self._build_loss(self.n_classes)
        pretrain_optimizer = self._build_optimizer(optimizer_class, optimizer_kwargs)
        self.pretrain_model.compile(loss=pretrain_loss_fn, optimizer=pretrain_optimizer, metrics=['accuracy'])

        loss_fn = self._build_loss(self.n_classes)
        optimizer = self._build_optimizer(optimizer_class, optimizer_kwargs)
        self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

    def _build_loss(self, n_classes):
        if n_classes > 2:
            return SparseCategoricalCrossentropy()
        return BinaryCrossentropy()

    def _build_optimizer(self, optimizer_class, optimizer_kwargs):
        if optimizer_class is None:
            return Adam()
        if optimizer_kwargs is None:
            return optimizer_class()
        return optimizer_class(**optimizer_kwargs)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Fit a model based on a stream batch."""
        X = sub_batch(X, self.sub_batch_size)
        y = sub_batch(to_categorical(y), self.sub_batch_size)

        if self.first_run:
            self._fit_model(self.pretrain_model, X, y, self.pretrain_n_epochs)

            self.model.set_weights(self.pretrain_model.get_weights())
            self.first_run = False

        else:
            self._fit_model(self.model, X, y, self.n_epochs)

        return self

    def _fit_model(self, model, X, y, n_epochs):
        model.fit(X, y, batch_size=X.shape[0], epochs=n_epochs, callbacks=self.callbacks, verbose=self.verbosity)
        if self.stateful:
            self._update_state(model)

    def predict(self, X):
        """Estimate predictions of `X` as binary values."""
        y_pred = self.predict_proba(X)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        """Estimate predictions of `X` as probabilities."""
        X = sub_batch(X, self.sub_batch_size)
        batch_size = X.shape[0]
        y_pred = self.model.predict(X, batch_size=batch_size)
        return unroll_predictions(y_pred)
