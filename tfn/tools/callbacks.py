import os
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import get as get_loss

from .converters import ndarrays_to_xyz
from ..layers import MaskedDistanceMatrix, OneHot


class TestModel(Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test

    def on_train_end(self, logs=None):
        predictions = self.model.evaluate(x=self.x_test, y=self.y_test)
        if not isinstance(predictions, list):
            predictions = [predictions]
        for pred, name in zip(predictions, self.model.metric_names):
            logs["test_{}".format(name)] = pred


class CartesianMetrics(Callback):
    def __init__(
        self,
        path: Union[str, Path],
        train: list = None,
        validation: list = None,
        test: list = None,
        max_structures: int = 64,
        write_rate: int = 50,
        tensorboard_logdir: str = None,
    ):
        """
        :param path: Base path to which .xyz files will be written to. Will create subdirectories
            at this path as needed.
        :param validation: Defaults to None. List of val data to write. Of shape [[z, r, p], [ts]].
            If None, will not write validation .xyz files.
        :param test: list. Defaults to None. Same as validation, but for test data.
        :param max_structures: int. Defaults to 10. Max number of structures to write .xyz files
            for.
        :param write_rate: int. Defaults to 50. Number of epochs to take before writing
            validation Cartesians (if validation data provided).
        """
        super().__init__()
        self.path = Path(path)
        self.train = train
        self.validation = validation
        self.test = test
        self.max_structures = max_structures
        self.write_rate = write_rate
        logdir = tensorboard_logdir or self.path.parent / "logs"
        self.file_writers = [
            tf.summary.create_file_writer(str(logdir / "train")),
            tf.summary.create_file_writer(str(logdir / "train")),
            tf.summary.create_file_writer(str(logdir / "validation")),
            tf.summary.create_file_writer(str(logdir / "validation")),
        ]

        self._prediction_type = "vectors"
        self._output_type = "cartesians"

    def get_vectors(self, inputs):
        model = Model(
            inputs=self.model.input, outputs=self.model.get_layer("vectors").output,
        )
        vectors = model.predict([np.expand_dims(a, axis=0) for a in inputs])
        return np.squeeze(vectors, axis=0)

    @staticmethod
    def write_vectors(vectors, path, i):
        vector_path = path / "vectors"
        os.makedirs(vector_path, exist_ok=True)
        np.savetxt(vector_path / f"{i}_vectors.txt", vectors)  # Write vectors .txt file

    def _get_prediction(self, x):
        if self._output_type == "distance_matrix":
            model = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("cartesians").output,
            )
        else:
            model = self.model
        return model.predict(x)

    def _unwrap_data_lazily(self, data: list):
        """
        :param data: data in format (x, y) where x is [atomic_nums, reactants, products] and y is
            [ts_cartesians].
        :return: i, z, r, p, ts_true, ts_pred
        """
        predicted_transition_states = self._get_prediction(data[0])
        ((atomic_nums, reactants, products), (true_transition_states,),) = data
        for i, (z, r, p, ts_true, ts_pred) in enumerate(
            zip(
                atomic_nums[: self.max_structures],
                reactants[: self.max_structures],
                products[: self.max_structures],
                true_transition_states[: self.max_structures],
                predicted_transition_states[: self.max_structures],
            )
        ):
            yield i, z, r, p, ts_true, ts_pred

    def loss(self, a, b):
        if a.shape != b.shape:
            return 0
        else:
            return np.mean(get_loss(self.model.loss)(a, b))

    @staticmethod
    def structure_loss(z, y_pred, y_true):
        d = MaskedDistanceMatrix()
        one_hot = OneHot(np.max(z) + 1)(z)
        dist_matrix = np.abs(d([one_hot, y_pred]) - d([one_hot, y_true]))
        dist_matrix = np.triu(dist_matrix)
        return (
            float(np.mean(dist_matrix[dist_matrix != 0])),
            float(np.mean(np.sum(np.sum(dist_matrix, axis=-1), axis=-1), axis=0)),
        )

    def write_cartesians(self, data: list, path: Path):
        """
        :param data: list. Of shape [[z, r, p], [true]]
        :param path: Path object. Base path subdirectories and .xyz files will be written under.
        """
        for i, z, r, p, ts, pred in self._unwrap_data_lazily(data):
            if self._prediction_type == "vectors":
                # Write vectors
                vectors = self.get_vectors([z, r, p])
                self.write_vectors(vectors, path, i)
            else:
                vectors = [0.0, 0.0]

            m = (r + p) / 2
            pred_message = (
                f"loss: {self.loss(pred, ts)}  "
                f"-- largest vector component: {np.max(vectors)} "
                f"-- smallest vector component {np.min(vectors)}"
            )
            # Write files
            if self._output_type != "distance_matrix":
                ndarrays_to_xyz(
                    ts, z, path / f"true/{i}_true.xyz", f"{self.loss(ts, ts)}"
                )
            ndarrays_to_xyz(
                pred, z, path / f"predicted/{i}_pred.xyz", pred_message,
            )
            ndarrays_to_xyz(
                m, z, path / f"midpoints/{i}_midpoint.xyz", f"{self.loss(m, ts)}",
            )
            ndarrays_to_xyz(
                r, z, path / f"reactants/{i}_reactant.xyz", f"{self.loss(r, ts)}"
            )
            ndarrays_to_xyz(
                p, z, path / f"products/{i}_product.xyz", f"{self.loss(p, ts)}"
            )

    def compute_metrics(self, epoch, split: str = "train"):
        if split == "train":
            data = self.train
            file_writers = self.file_writers[:2]
        else:
            data = self.validation
            file_writers = self.file_writers[2:]
        metrics = {
            name: metric
            for name, metric in zip(
                [f"mean_distance_error", f"manhattan_distance_error",],
                self._compute_metrics(data),
            )
        }
        self.write_metrics(metrics, epoch, file_writers, split)

    def _compute_metrics(self, data):
        z = data[0][0]
        y_pred = data[1][0]
        y_true = self.model.predict(data[0])
        return self.structure_loss(z, y_pred, y_true)

    def _writing(self, epoch):
        return epoch == 0 or (epoch + 1) % self.write_rate == 0

    def write_metrics(
        self,
        metrics: dict,
        epoch: int,
        file_writers: list = None,
        prefix: str = "scalar",
    ):
        file_writers = file_writers or self.file_writers
        print(
            " -- ".join(
                [
                    f"{prefix}_{name}: {round(metric, 4)}"
                    for name, metric in metrics.items()
                ]
            )
        )
        for writer, (name, metric) in zip(file_writers, metrics.items()):
            with writer.as_default():
                tf.summary.scalar(name, metric, epoch)

    def on_train_begin(self, logs=None):
        data = self.validation or self.test
        if data is None:
            return
        else:
            if data[1][-1].shape[-1] != 3:
                self._output_type = "distance_matrix"
            if "vectors" not in [layer.name for layer in self.model.layers]:
                self._prediction_type = "cartesians"

    def on_epoch_end(self, epoch, logs=None):
        if self.validation is None or self.train is None:
            return
        else:
            if self._output_type == "cartesians":
                self.compute_metrics(epoch, "train")
                self.compute_metrics(epoch, "val")

            if self._writing(epoch):
                (train_z, train_r, train_p), (train_ts,) = self.train
                (val_z, val_r, val_p), (val_ts,) = self.validation

                train_midpoint_loss = self.loss((train_r + train_p) / 2, train_ts)
                val_midpoint_loss = self.loss((val_r + val_p) / 2, val_ts)
                tf.summary.scalar("val_midpoint_loss", val_midpoint_loss, epoch)
                tf.summary.scalar("train_midpoint_loss", train_midpoint_loss, epoch)

                self.write_cartesians(
                    self.validation, self.path / f"val/epoch_{epoch + 1}"
                )
                self.write_cartesians(
                    self.train, self.path / f"train/epoch_{epoch + 1}"
                )

    def on_train_end(self, logs=None):
        if self.test is None:
            return
        else:
            (z, r, p), (ts,) = self.validation
            midpoint_loss = self.loss((r + p) / 2, ts)
            print(f"midpoint test loss: {midpoint_loss}")
            self.write_cartesians(self.train, self.path / "train/post_training")
            self.write_cartesians(self.validation, self.path / "val/post_training")
            self.write_cartesians(self.test, self.path / "test")


class ClassificationMetrics(Callback):
    def __init__(self, validation, log_dir):
        super().__init__()
        self.validation = validation
        self.val_f1s = None
        self.val_recalls = None
        self.val_precisions = None
        self.file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        self.file_writer.set_as_default()

    def on_epoch_end(self, epoch, logs=None):
        target = self.validation[1]
        prediction = np.asarray(self.model.predict(self.validation[0]))
        f1score = self.f1_score(target, prediction)
        precision = self.precision(target, prediction)
        recall = self.recall(target, prediction)
        accuracy = np.mean(categorical_accuracy(target, prediction))
        tf.summary.scalar("f1score", f1score, epoch)
        tf.summary.scalar("precision", precision, epoch)
        tf.summary.scalar("recall", recall, epoch)
        tf.summary.scalar("accuracy", accuracy, epoch)
        print(
            f"Metrics for epoch {epoch}:"
            f" -- val_f1score: {f1score} -- val_precision: {precision} -- val_recall: {recall} "
            f" -- val_accuracy: {accuracy}"
        )

    def f1_score(self, y_true, y_pred):
        recall = self.recall(y_true, y_pred)
        precision = self.precision(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + 1e-7))

    def recall(self, y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + 1e-7)

    def precision(self, y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + 1e-7)
