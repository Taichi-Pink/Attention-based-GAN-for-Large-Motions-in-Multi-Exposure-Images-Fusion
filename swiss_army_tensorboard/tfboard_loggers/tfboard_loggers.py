import re
import warnings
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import Session


class _BaseTFBoardLogger:
    def __init__(self, log_dir: Union[Path, str]):
        log_dir = str(log_dir)

        if Path(log_dir).is_dir():
            warnings.warn("Folder {0} is already created, maybe it contains other log files".format(log_dir))

        self._log_dir = log_dir
        self._summary_writer = tf.summary.FileWriter(log_dir)


class TFBoardModelGraphLogger:
    def __init__(self):
        raise NotImplemented("You don't need to use the constructor of this class")

    @staticmethod
    def log_graph(log_dir, model_session: Session):
        # You can get the session by: keras.backend.get_session()
        _ = tf.summary.FileWriter(str(log_dir), model_session.graph)


class TFBoardHistogramLogger(_BaseTFBoardLogger):
    def __init__(self, log_dir: Union[Path, str]):
        super().__init__(log_dir)

    def log_histogram(self, tag: str, values: Union[np.ndarray, list], step: int, bins: int):
        values = np.array(values)
        histogram = self._create_histogram(values, bins)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=histogram)])
        self._summary_writer.add_summary(summary, step)
        self._summary_writer.flush()

    @staticmethod
    def _create_histogram(values: np.ndarray, bins: int) -> tf.HistogramProto:
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        return hist


class TFBoardScalarLogger(_BaseTFBoardLogger):
    def __init__(self, log_dir: Union[Path, str]):
        super().__init__(log_dir)

    def log_scalar(self, tag: str, value, step: int):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._summary_writer.add_summary(summary, step)


class TFBoardImageLogger(_BaseTFBoardLogger):
    def __init__(self, log_dir: Union[Path, str]):
        super().__init__(log_dir)

    def log_images(self, tag: str, images: Union[List[np.ndarray], np.ndarray], step: int):
        image_summaries = []
        for i, image in enumerate(images):
            image_str = BytesIO()
            plt.imsave(image_str, image, format='png')
            height, width = image.shape[:2]
            image_summary = tf.Summary.Image(encoded_image_string=image_str.getvalue(),
                                             height=height,
                                             width=width)
            image_tag = "{0}/{1}".format(tag, i)
            image_summaries.append(tf.Summary.Value(tag=image_tag, image=image_summary))

        summary = tf.Summary(value=image_summaries)
        self._summary_writer.add_summary(summary, step)


class TFBoardTextLogger(_BaseTFBoardLogger):
    def __init__(self, log_dir: Union[Path, str]):
        super().__init__(log_dir)

    def log_markdown(self, tag: str, text_in_markdown: str, step: int):
        text_tensor = tf.make_tensor_proto(text_in_markdown, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
        self._summary_writer.add_summary(summary, step)

    def log_text(self, tag: str, text: str, step: int):
        text = self._convert_simple_text_to_markdown(text)
        self.log_markdown(tag, text, step)

    @staticmethod
    def _convert_simple_text_to_markdown(text: str) -> str:
        # TODO: real conversion!
        # For example: for every newline we need to insert another newline (\n\n), so when it gets rendered in markdown
        # it will be seen as a newline
        return text.replace("\n", "\n\n")


class TFBoardContinuousTextLogger:
    # TODO: use enum
    _ACCEPTED_LOGGER_STYLES = ["normal", "code"]

    def __init__(self, log_dir: Union[Path, str], tag: str, logger_style: str = "normal", max_line_numbers: int = 500,
                 step: int = -1):
        self._step = step
        self._tag = tag

        self._logger_style = logger_style.lower()
        if self._logger_style not in self._ACCEPTED_LOGGER_STYLES:
            raise ValueError(
                "Logger style should be one of the following: {0}".format(" ".join(self._ACCEPTED_LOGGER_STYLES)))

        self._logged_texts = deque([""], maxlen=max_line_numbers)
        self._styling_dict = {"info": "", "warning": "*", "error": "**"}

        self._text_logger = TFBoardTextLogger(log_dir)

    def _log_with_formatting(self, text: str, formatting_character_before: str = "",
                             formatting_character_after: str = ""):
        formatted_text = text
        if self._logger_style != "code":
            formatted_text = formatting_character_before + text + formatting_character_after
        self._logged_texts.append(formatted_text)

    def info(self, text: str):
        f_char = self._styling_dict["info"]
        self._log_with_formatting(text, f_char, f_char)

    def error(self, text: str):
        f_char = self._styling_dict["error"]
        self._log_with_formatting(text, f_char, f_char)

    def warn(self, text: str):
        f_char = self._styling_dict["warning"]
        self._log_with_formatting(text, f_char, f_char)

    def markdown(self, text: str):
        self._log_with_formatting(text, "", "")

    def _construct_markdown_logger_text(self):
        if self._logger_style == "code":
            # 2 line breaks and 4 white spaces
            markdown_text = "\n\n    ".join(self._logged_texts)
        elif self._logger_style == "normal":
            markdown_text = "\n\n".join(self._logged_texts)
        else:
            markdown_text = "\n\n".join(self._logged_texts)

        return markdown_text

    def stop_logging(self):
        text = self._construct_markdown_logger_text()
        self._text_logger.log_markdown(self._tag, text, self._step)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop_logging()


class TFBoardKerasModelWeightsLogger(TFBoardHistogramLogger):
    def __init__(self, log_dir: Union[Path, str]):
        super().__init__(log_dir)

    def log_weights(self, model, step: int, layer_regex: str = ".*", hist_bins: int = 100):
        for layer in model.layers:
            if bool(re.fullmatch(layer_regex, layer.name)):
                if not layer.weights:
                    continue
                for weight, weights_numpy_array in zip(layer.weights, layer.get_weights()):
                    weights_name = weight.name.replace(":", "_")
                    self.log_histogram(weights_name, weights_numpy_array, step, bins=hist_bins)
