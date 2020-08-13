from sacred.run import Run
from tensorflow.keras.callbacks import Callback


class SacredMetricLogger(Callback):
    def __init__(self, _run: Run):
        super().__init__()
        self.run = _run
        self.batch_log_rate = 500
        self.epoch_log_rate = 1

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch % self.epoch_log_rate == 0:
            for key, value in logs.items():
                if key != "epoch" and key != "size":
                    self.run.log_scalar("{}".format(key), value=value, step=epoch)
                    if epoch % (self.epoch_log_rate * 10) == 0:
                        self.run.result = value
