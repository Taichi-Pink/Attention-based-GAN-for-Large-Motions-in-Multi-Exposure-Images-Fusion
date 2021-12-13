import os
import shutil
from typing import Union

import tensorflow as tf

from swiss_army_tensorboard import tfboard_loggers


class TFBoardTrainValidationLossCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir: str, sub_dir_training="training_losses", sub_dir_validation="validation_losses",
                 remove_already_existing_folders: bool = False, write_graph: bool = True,
                 update_freq: Union[str, int] = "epochs"):
        self.training_log_dir = os.path.join(log_dir, sub_dir_training)
        self.val_log_dir = os.path.join(log_dir, sub_dir_validation)

        if remove_already_existing_folders:
            if os.path.isdir(self.training_log_dir):
                shutil.rmtree(self.training_log_dir)

            if os.path.isdir(self.val_log_dir):
                shutil.rmtree(self.val_log_dir)

        super().__init__(log_dir=self.training_log_dir, write_graph=write_graph, update_freq=update_freq)

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super().set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.val_writer.close()


class TFBoardWeightHistogramsCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer_regex: str, log_dir: str, sub_dir="histograms", hist_bins: int = 100,
                 remove_already_existing_folders: bool = False):
        super().__init__()
        self.layer_regex = layer_regex
        self.hist_bins = hist_bins

        histogram_log_dir = os.path.join(log_dir, sub_dir)

        if remove_already_existing_folders:
            if os.path.isdir(histogram_log_dir):
                shutil.rmtree(histogram_log_dir)

        self.weights_logger = tfboard_loggers.TFBoardKerasModelWeightsLogger(histogram_log_dir)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.weights_logger.log_weights(self.model, epoch, self.layer_regex, self.hist_bins)
