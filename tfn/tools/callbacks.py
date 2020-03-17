from tensorflow.python.keras.callbacks import Callback


class TestModel(Callback):
    def __init__(self,
                 x_test,
                 y_test):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test

    def on_train_end(self, logs=None):
        predictions = self.model.evaluate(x=self.x_test, y=self.y_test)
        if not isinstance(predictions, list):
            predictions = [predictions]
        for pred, name in zip(predictions, self.model.metric_names):
            logs['test_{}'.format(name)] = pred
