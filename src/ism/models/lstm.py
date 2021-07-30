from tensorflow.keras.layers import LSTM

from ism.core.model import IncrementalRecurrentClassifier


class IncrementalLSTMClassifier(IncrementalRecurrentClassifier):
    """Incremental LSTM classifier with scikit-flow API and based on Tensorflow/Keras API.

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

    Examples:
        >>> from ism.models import IncrementalLSTMClassifier
        >>> from skmultiflow.evaluation import EvaluatePrequential
        >>> from skmultiflow.data.sine_generator import SineGenerator
        >>> 
        >>> PRETRAIN_BATCH_SIZE = 100
        >>> BATCH_SIZE = 100
        >>> 
        >>> ilstm = IncrementalLSTMClassifier(
        ...     n_classes=2,
        ...     n_features=2,
        ...     pretrain_n_epochs=600,
        ...     n_epochs=300,
        ...     recurrent_layers_spec=[
        ...         dict(units=150, activation='tanh'),
        ...         dict(units=100, activation='tanh'),
        ...         dict(units=50, activation='tanh')
        ...     ],
        ...     pretrain_batch_size=BATCH_SIZE,
        ...     batch_size=BATCH_SIZE,
        ...     sub_batch_size=50,
        ...     stateful=True
        ... )
        >>> 
        >>> evaluator = EvaluatePrequential(
        ...     pretrain_size=PRETRAIN_BATCH_SIZE,
        ...     batch_size=BATCH_SIZE,
        ...     max_samples=1000,
        ...     metrics=['accuracy']
        ... )
        >>> evaluator.evaluate(
        ...     stream=SineGenerator(random_state=2021),
        ...     model=[ilstm],
        ...     model_names=['ILSTM']
        ... )
    """
    @property
    def recurrrent_layer_class(self):
        """A sub-class of `tensorflow.keras.layers.RNN`."""
        return LSTM
