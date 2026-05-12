import json
import re
import typing as tp
from rapidfuzz.distance import Levenshtein

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.registry import DICT_POSTPROCESSORS
from opencompass.utils import get_data_path
from opencompass.openicl import BaseEvaluator

from .base import BaseDataset


@LOAD_DATASET.register_module()
class Text2JSONDataset(BaseDataset):
    @staticmethod
    def load(path, n_samples, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                dataset.append(line)
                if i == n_samples - 1:
                    break
        return Dataset.from_list(dataset)

@DICT_POSTPROCESSORS.register_module()
def text2json_llm_eval_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    scores = []
    for key in output.keys():
        pred = output[key]["prediction"]
        res = re.search(r'"score.*(\d\d)', pred)
        if res is not None:
            score = float(res.group(1))
            scores.append(score)
    score = sum(scores) / len(scores)
    result = {
        "score": score,
        # "details": {"scores": scores, "num_scored": len(scores), "output": output},
    }

    return result


class Text2jsonEvaluator(BaseEvaluator):
    def score(self, predictions: tp.List[str], gold: tp.List[str]) -> tp.Dict[str, tp.Any]:
        assert len(predictions) == len(gold), (len(predictions), len(gold))
        n_samples = len(predictions)
        total_score = 0
        bad_samples = 0
        for i, (p, g) in enumerate(zip(predictions, gold)):
            g_json: tp.List[tp.Dict[str, str]] = json.loads(g)
            assert type(g_json) == list, type(g_json)
            gold_names_mapping: tp.Dict[str, tp.Dict[str, str]] = dict()
            break_because_of_bad_sample = False
            for instance in g_json:
                assert len(instance.keys()) == 3
                name = instance['name'].replace("’", "'")
                del instance['name']
                if name in gold_names_mapping:
                    bad_samples += 1
                    break_because_of_bad_sample = True
                    break
                gold_names_mapping[name] = instance
            if break_because_of_bad_sample:
                continue
            max_instance_score = 1 # 100 / len(gold_names_mapping)
            missing_key_penalty = max_instance_score / 2 # 2 because there are 3 keys: name and 2 other keys
            wrong_value_penalty = max_instance_score / 4 # just half of missing key penalty
            union_instances = len(gold_names_mapping)
            if p.count("[") != 1 or p.count("]") != 1 or p.index("[") > p.index("]"):
                continue
            p = p[p.index('['): p.index(']') + 1]
            try:
                prediction_json = json.loads(p)
            except Exception: 
                continue
            if type(prediction_json) != list:
                continue
            sample_score = 0
            for instance in prediction_json:
                if 'name' not in instance or type(instance['name']) != str:
                    union_instances += 1
                    continue
                name = instance['name'].replace("’", "'")
                if name not in gold_names_mapping:
                    union_instances += 1
                    continue
                instance_score = max_instance_score
                for key, value in gold_names_mapping[name].items():
                    if key not in instance or instance[key] is None:
                        instance_score -= missing_key_penalty
                    else:
                        assert type(value) == str
                        if type(instance[key]) == int:
                            instance[key] = str(instance[key])
                        value = value.replace("’", "'")
                        instance[key] = instance[key].replace("’", "'")
                        if value != instance[key]:
                            instance_score -= wrong_value_penalty
                sample_score += instance_score
                del gold_names_mapping[name]
            total_score += sample_score / union_instances
        return {
            # 'details': f"bad_sampels={bad_samples}, broken_json_generated={broken_json_generated}, wrong_json_type_generated={wrong_json_type_generated}",
            'average_score': 100 * total_score / (n_samples - bad_samples),
        }

class Text2jsonOrderedEvaluator(BaseEvaluator):
    def score(self, predictions: tp.List[str], gold: tp.List[str]) -> tp.Dict[str, tp.Any]:
        assert len(predictions) == len(gold), (len(predictions), len(gold))
        errors = 0
        n_entries = 0
        for i, (p, g) in enumerate(zip(predictions, gold)):
            g_json: tp.List[tp.Dict[str, str]] = json.loads(g)
            assert type(g_json) == list, type(g_json)
            keys = set(g_json[0].keys())
            for instance in g_json[1:]:
                assert set(instance.keys()) == keys
            fixed_keys_order = list(keys)
            try:
                p_json = json.loads(p)
            except json.JSONDecodeError:
                print(i, "broken json")
                continue
            if type(p_json) != list:
                print(i, "not a list")
                continue
            
            predicted_strings, gold_strings = [], []
            for i, instance in enumerate(g_json):
                gold_strings.append("$".join(instance[k] for k in fixed_keys_order))
            for i, instance in enumerate(p_json):
                extra_keys = set(instance.keys()) - keys
                predicted_string = "$".join([instance.get(k, "[{(---NONE---)}]") for k in fixed_keys_order] + [instance[k] for k in extra_keys])
                predicted_strings.append(predicted_string)

            dist = Levenshtein.distance(gold_strings, predicted_strings)
            errors += dist
            n_entries += len(gold_strings)
        
        return {
            'average_score': 1 - errors / n_entries
        }
