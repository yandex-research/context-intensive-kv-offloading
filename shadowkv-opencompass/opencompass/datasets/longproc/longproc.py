import typing as tp
from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
from .longproc_data import load_longproc_data
from opencompass.openicl import BaseEvaluator
from collections import defaultdict

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class LongProcDataset(BaseDataset):
    @staticmethod
    def load(path, dataset_name, **kwargs):
        path = get_data_path(path)
        data, _ = load_longproc_data(dataset_name, path)
        return Dataset.from_list(data)

class LongProcEvaluator(BaseEvaluator):
    def __init__(self, path, dataset_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = get_data_path(path)
        _, self.eval_func = load_longproc_data(dataset_name, path)

    def score(self, predictions: tp.List[str], gold: tp.List[str]) -> tp.Dict[str, tp.Any]:
        total_metrics = defaultdict(list)

        for prediction, g in zip(predictions, gold):
            metrics, _ = self.eval_func(prediction, {"reference_output": g})
            for metric, value in metrics.items():
                total_metrics[metric].append(value)

        return {key: sum(values) / len(values) * 100 for key, values in total_metrics.items()}
