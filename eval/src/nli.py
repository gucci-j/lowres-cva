# ruff: noqa: F405, F403, F401
"""
NLI evaluation tasks for lighteval.
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


task_ja = LightevalTaskConfig(
    name="nli:ja",
    prompt_function="nli_prompt_fn",  
    hf_repo="your-hf-id/nli-ja", # TODO: Need to change here
    hf_subset="default",
    metric=["loglikelihood_acc"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=1,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_de = LightevalTaskConfig(
    name="nli:de",
    prompt_function="nli_prompt_fn",  
    hf_repo="your-hf-id/nli-de", # TODO: Need to change here
    hf_subset="default",
    metric=["loglikelihood_acc"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=1,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_ar = LightevalTaskConfig(
    name="nli:ar",
    prompt_function="nli_prompt_fn",  
    hf_repo="your-hf-id/nli-ar", # TODO: Need to change here
    hf_subset="default",
    metric=["loglikelihood_acc"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=1,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_sw = LightevalTaskConfig(
    name="nli:sw",
    prompt_function="nli_prompt_fn",  
    hf_repo="your-hf-id/nli-sw", # TODO: Need to change here
    hf_subset="default",
    metric=["loglikelihood_acc"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=1,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_th = LightevalTaskConfig(
    name="nli:th",
    prompt_function="nli_prompt_fn",  
    hf_repo="your-hf-id/nli-th", # TODO: Need to change here
    hf_subset="default",
    metric=["loglikelihood_acc"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=1,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_hi = LightevalTaskConfig(
    name="nli:hi",
    prompt_function="nli_prompt_fn",  
    hf_repo="your-hf-id/nli-hi", # TODO: Need to change here
    hf_subset="default",
    metric=["loglikelihood_acc"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=1,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_el = LightevalTaskConfig(
    name="nli:el",
    prompt_function="nli_prompt_fn",  
    hf_repo="your-hf-id/nli-el", # TODO: Need to change here
    hf_subset="default",
    metric=["loglikelihood_acc"], 
    hf_avail_splits=["train", "test"], 
    evaluation_splits=["test"], 
    few_shots_split="train", 
    few_shots_select="balanced",
    suite=["custom"],
    generation_size=1,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

def nli_prompt_fn(
    line, 
    task_name: str = None
):
    premise = line["premise"]
    hypothesis = line["hypothesis"]
    lang_code = task_name.split(":")[1]

    if lang_code == "ja":
        return Doc(
            task_name=task_name,
            query=f"{premise} 質問: {hypothesis} 真、偽、どちらでもない？ 答え:",
            choices=[" 真", " どちらでもない", " 偽"],
            gold_index=int(line["label"]),
        )
    
    elif lang_code == "de":
        return Doc(
            task_name=task_name,
            query=f"{premise} Frage: {hypothesis} Wahr, Falsch oder Weder? Antwort:",
            choices=[" Wahr", " Weder", " Falsch"],
            gold_index=int(line["label"]),
        )

    elif lang_code == "ar":
        return Doc(
            task_name=task_name,
            query=f"{premise} سؤال: {hypothesis} صحيح ، خطأ أو لا هذا ولا ذاك؟ إجابة:",
            choices=[" صحيح", " لا هذا ولا ذاك", " خطأ"],
            gold_index=int(line["label"]),
        )

    elif lang_code == "sw":
        return Doc(
            task_name=task_name,
            query=f"{premise} Swali: {hypothesis} Kweli, Uongo au Wala? Jibu:",
            choices=[" Kweli", " Wala", " Uongo"],
            gold_index=int(line["label"]),
        )

    elif lang_code == "th":
        return Doc(
            task_name=task_name,
            query=f"{premise} คำถาม: {hypothesis} จริง, เท็จ, ไม่แน่ใจ? คำตอบ:",
            choices=[" จริง", " ไม่แน่ใจ", " เท็จ"],
            gold_index=int(line["label"]),
        )

    elif lang_code == "hi":
        return Doc(
            task_name=task_name,
            query=f"{premise} प्रश्न: {hypothesis} सही, ना तो सही ना गलत, गलत? उत्तर:",
            choices=[" सही", " ना तो सही ना गलत", " गलत"],
            gold_index=int(line["label"]),
        )

    elif lang_code == "el":
        return Doc(
            task_name=task_name,
            query=f"{premise} Ερώτηση: {hypothesis} Αληθές, Ψευδές, ή Κανένα από τα δύο; Απάντηση:",
            choices=[" Αληθές", " Κανένα από τα δύο", " Ψευδές"],
            gold_index=int(line["label"]),
        )
    
    else:
        raise ValueError(f"Language code {lang_code} is not supported.")

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
