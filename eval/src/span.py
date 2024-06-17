# ruff: noqa: F405, F403, F401
"""
SPAN evaluation tasks for lighteval.
"""

import nltk
import numpy as np
from aenum import extend_enum

from lighteval.metrics import Metrics
from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.metrics_sample import F1_score, ExactMatches
from lighteval.metrics.normalizations import helm_normalizer
from lighteval.metrics.utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

task_ja = LightevalTaskConfig(
    name="span:ja",
    prompt_function="span_prompt_fn",  
    hf_repo="your-hf-id/span-ja", # TODO: Need to change here
    hf_subset="default",
    metric=["f1_score_quasi_ja", "quasi_exact_match_ja"], # TODO: Add specific metrics
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=8,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_de = LightevalTaskConfig(
    name="span:de",
    prompt_function="span_prompt_fn",  
    hf_repo="your-hf-id/span-de-truncated", # TODO: Need to change here
    hf_subset="default",
    metric=["f1_score_quasi", "quasi_exact_match"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=8,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_ar = LightevalTaskConfig(
    name="span:ar",
    prompt_function="span_prompt_fn",  
    hf_repo="your-hf-id/span-ar", # TODO: Need to change here
    hf_subset="default",
    metric=["f1_score_quasi", "quasi_exact_match"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=8,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_sw = LightevalTaskConfig(
    name="span:sw",
    prompt_function="span_prompt_fn",  
    hf_repo="your-hf-id/span-sw", # TODO: Need to change here
    hf_subset="default",
    metric=["f1_score_quasi", "quasi_exact_match"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=8,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_th = LightevalTaskConfig(
    name="span:th",
    prompt_function="span_prompt_fn",  
    hf_repo="your-hf-id/span-th", # TODO: Need to change here
    hf_subset="default",
    metric=["f1_score_quasi_th", "quasi_exact_match_th"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=8,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_hi = LightevalTaskConfig(
    name="span:hi",
    prompt_function="span_prompt_fn",  
    hf_repo="your-hf-id/span-hi", # TODO: Need to change here
    hf_subset="default",
    metric=["f1_score_quasi_hi", "quasi_exact_match_hi"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=8,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_el = LightevalTaskConfig(
    name="span:el",
    prompt_function="span_prompt_fn",  
    hf_repo="your-hf-id/span-el", # TODO: Need to change here
    hf_subset="default",
    metric=["f1_score_quasi", "quasi_exact_match"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=8,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

def span_prompt_fn(
    line, 
    task_name: str = None
):
    context = line["context"]
    question = line["question"]
    answer = line["answers"]["text"][0]
    lang_code = task_name.split(":")[1]

    if lang_code == "ja":
        return Doc(
            task_name=task_name,
            query=f"次の文章の質問に答えなさい。文章: {context} 質問: {question} 答え: ",
            gold_index=0,
            choices=[answer],
        )
    
    elif lang_code == "de":
        return Doc(
            task_name=task_name,
            query=f"Beantworten Sie die folgende Frage. Artikel: {context} Frage: {question} Antwort: ",
            gold_index=0,
            choices=[answer],
        )

    elif lang_code == "ar":
        return Doc(
            task_name=task_name,
            query=f"أجب على السؤال التالي. سياق: {context} السؤال: {question} الإجابة: ",
            gold_index=0,
            choices=[answer],
        )

    elif lang_code == "sw":
        return Doc(
            task_name=task_name,
            query=f"Jibu swali lifuatalo. Makala: {context} Swali: {question} Jibu: ",
            gold_index=0,
            choices=[answer],
        )

    elif lang_code == "th":
        return Doc(
            task_name=task_name,
            query=f"ตอบคำถามอันต่อไปนี้ บทความ: {context} คำถาม: {question} คำตอบ: ",
            gold_index=0,
            choices=[answer],
        )

    elif lang_code == "hi":
        return Doc(
            task_name=task_name,
            query=f"इस प्रश्न का उत्तर दें। संदर्भ: {context} प्रश्न: {question} उत्तर: ",
            gold_index=0,
            choices=[answer],
        )

    elif lang_code == "el":
        return Doc(
            task_name=task_name,
            query=f"Απάντησε στην παρακάτω ερώτηση. Κείμενο: {context} Ερώτηση: {question} Απάντηση: ",
            gold_index=0,
            choices=[answer],
        )
    
    else:
        raise ValueError(f"Language code {lang_code} is not supported.")



# CUSTOM METRIC IF NEEDED
class F1_score_ja(F1_score):
    """F1 score for Japanese."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def compute_one_item(self, gold: str, pred: str) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The f1 score over the bag of words, computed using nltk.
        """
        # preprocessing
        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)
        gold = " ".join(gold.replace(" ", "").rstrip("。"))
        pred = " ".join(pred.replace(" ", "").rstrip("。"))
        gold_bow = set(gold.split())
        pred_bow = set(pred.split())

        # compute f1
        ret = nltk.scores.f_measure(gold_bow, pred_bow)

        if ret is None:
            return 0.0
        return ret

f1_score_quasi_ja = SampleLevelMetric(
    metric="f1_score_quasi_ja",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=F1_score_ja(
        normalize_gold=helm_normalizer, 
        normalize_pred=helm_normalizer).compute,  # how to compute score for one sample
    corpus_level_fn=np.mean,  # aggregation
)
extend_enum(Metrics, "f1_score_quasi_ja", f1_score_quasi_ja)


class ExactMatches_ja(ExactMatches):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The exact match score. Will be 1 for a match, 0 otherwise.
        """
        if not pred:
            return 0

        # preprocessing
        if self.strip_strings:
            gold = gold.strip()
            pred = pred.strip()
        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)
        gold = " ".join(gold.replace(" ", "").rstrip("。"))
        pred = " ".join(pred.replace(" ", "").rstrip("。"))

        # compute exact match
        if self.type_exact_match == "prefix":
            return 1 if pred.startswith(gold) else 0
        if self.type_exact_match == "suffix":
            return 1 if pred.endswith(gold) else 0
        return 1 if gold == pred else 0


quasi_exact_match_ja = SampleLevelMetric(
    metric="qem_ja",
    sample_level_fn=ExactMatches_ja(
        normalize_gold=helm_normalizer,
        normalize_pred=helm_normalizer,
        strip_strings=True,
    ).compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
extend_enum(Metrics, "quasi_exact_match_ja", quasi_exact_match_ja)


class F1_score_th(F1_score):
    """F1 score for Thai."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def compute_one_item(self, gold: str, pred: str) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The f1 score over the bag of words, computed using nltk.
        """
        # preprocessing
        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)
        gold = " ".join(gold.replace(" ", ""))
        pred = " ".join(pred.replace(" ", ""))
        gold_bow = set(gold.split())
        pred_bow = set(pred.split())

        # compute f1
        ret = nltk.scores.f_measure(gold_bow, pred_bow)

        if ret is None:
            return 0.0
        return ret

f1_score_quasi_th = SampleLevelMetric(
    metric="f1_score_quasi_th",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=F1_score_th(
        normalize_gold=helm_normalizer, 
        normalize_pred=helm_normalizer).compute,  # how to compute score for one sample
    corpus_level_fn=np.mean,  # aggregation
)
extend_enum(Metrics, "f1_score_quasi_th", f1_score_quasi_th)


class ExactMatches_th(ExactMatches):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The exact match score. Will be 1 for a match, 0 otherwise.
        """
        if not pred:
            return 0

        # preprocessing
        if self.strip_strings:
            gold = gold.strip()
            pred = pred.strip()
        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)
        gold = " ".join(gold.replace(" ", ""))
        pred = " ".join(pred.replace(" ", ""))

        # compute exact match
        if self.type_exact_match == "prefix":
            return 1 if pred.startswith(gold) else 0
        if self.type_exact_match == "suffix":
            return 1 if pred.endswith(gold) else 0
        return 1 if gold == pred else 0


quasi_exact_match_th = SampleLevelMetric(
    metric="qem_th",
    sample_level_fn=ExactMatches_th(
        normalize_gold=helm_normalizer,
        normalize_pred=helm_normalizer,
        strip_strings=True,
    ).compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
extend_enum(Metrics, "quasi_exact_match_th", quasi_exact_match_th)


class F1_score_hi(F1_score):
    """F1 score for Hindi."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def compute_one_item(self, gold: str, pred: str) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The f1 score over the bag of words, computed using nltk.
        """
        # preprocessing
        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)
        gold = " ".join(gold.replace(" ", ""))
        pred = " ".join(pred.replace(" ", ""))
        gold_bow = set(gold.split())
        pred_bow = set(pred.split())

        # compute f1
        ret = nltk.scores.f_measure(gold_bow, pred_bow)

        if ret is None:
            return 0.0
        return ret

f1_score_quasi_hi = SampleLevelMetric(
    metric="f1_score_quasi_hi",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=F1_score_hi(
        normalize_gold=helm_normalizer, 
        normalize_pred=helm_normalizer).compute,  # how to compute score for one sample
    corpus_level_fn=np.mean,  # aggregation
)
extend_enum(Metrics, "f1_score_quasi_hi", f1_score_quasi_hi)


class ExactMatches_hi(ExactMatches):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The exact match score. Will be 1 for a match, 0 otherwise.
        """
        if not pred:
            return 0

        # preprocessing
        if self.strip_strings:
            gold = gold.strip()
            pred = pred.strip()
        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)
        gold = " ".join(gold.replace(" ", ""))
        pred = " ".join(pred.replace(" ", ""))

        # compute exact match
        if self.type_exact_match == "prefix":
            return 1 if pred.startswith(gold) else 0
        if self.type_exact_match == "suffix":
            return 1 if pred.endswith(gold) else 0
        return 1 if gold == pred else 0


quasi_exact_match_hi = SampleLevelMetric(
    metric="qem_hi",
    sample_level_fn=ExactMatches_hi(
        normalize_gold=helm_normalizer,
        normalize_pred=helm_normalizer,
        strip_strings=True,
    ).compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
extend_enum(Metrics, "quasi_exact_match_hi", quasi_exact_match_hi)


# MODULE LOGIC
_TASKS = (
    task_ja,
    task_de,
    task_ar,
    task_sw,
    task_th,
    task_hi,
    task_el,
)
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
