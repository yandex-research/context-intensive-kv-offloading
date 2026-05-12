import os

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.text2json import Text2JSONDataset, Text2jsonEvaluator, Text2jsonOrderedEvaluator
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets.text2json import text2json_llm_eval_postprocess
from opencompass.models import HuggingFacewithChatTemplate

text2json_reader_cfg = dict(input_columns=["problem"], output_column="answer")


text2json_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="{problem}",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    # \underset{i \in [1, 2, ..., 500]} \max tokenize(gold[i]).shape[-1] = 905, so 1024 must be  enough
    # and it will make experiments much faster, because EOS is not generated in half of the samples with the default ShadowKV parameters
    inferencer=dict(type=GenInferencer, max_out_len=1024), 
)

GRADER_TEMPLATE = """s
You are an automated JSON-evaluation judge. Your task is to compare an **Assistant JSON** to a **Ground Truth JSON** and produce a **single overall similarity score from 1 to 100** (higher is better), plus a brief justification.

### Inputs
- **GROUND_TRUTH_JSON**: `{GROUND_TRUTH_JSON}`
- **ASSISTANT_JSON**: `{ASSISTANT_JSON}`

### Evaluation rules
1. **Parse and validate JSON**
- If ASSISTANT_JSON is not valid JSON, score **1**.
- Treat object key order as irrelevant.

2. **Compare structure and schema (high weight)**
- Do the top-level types match (object/array)?
- Are required keys present?
- Are nesting levels and container types (object vs array) correct?
- Penalize missing keys and incorrect nesting strongly.

3. **Compare fields and values (high weight)**
- For each key present in ground truth, check:
    - **Type match** (string/number/bool/null/object/array)
    - **Value match**
- Exact matches score best.
- Minor formatting differences (whitespace, punctuation) should receive small/no penalty if meaning is unchanged.
- Do not penalize for differences in indentation.
- If values are numeric, allow small tolerance only if clearly intended (e.g., rounding). Otherwise treat as mismatch.

4. **Arrays**
- If the array is **order-insensitive by nature** (e.g., set-like lists), score using best matching regardless of order.
- If order is semantically meaningful (steps, ranked results), penalize order mismatches.
- Missing/extra elements should be penalized proportional to importance and count.

5. **Extra keys**
- Extra keys in ASSISTANT_JSON not present in ground truth should incur a penalty.
- If extras are clearly harmless metadata and do not conflict, apply a smaller penalty.

6. **Semantic equivalence**
- If two strings differ but are clearly equivalent in meaning (synonyms, normalized formatting like dates `"2026-02-24"` vs `"2026/02/24"`), do not penalize.
- Allow differences in spelling, punctuation, and case.

### Scoring rubric (use this as guidance)
- **95–100**: Essentially identical; only trivial differences.
- **85–94**: Very close; small number of minor issues (small missing/extra fields or slight value differences).
- **70–84**: Mostly correct structure; several mismatched values or a few structural issues.
- **50–69**: Partial match; significant missing/incorrect fields; structure somewhat off.
- **20–49**: Major structural/value divergence; only a small subset matches.
- **1–19**: Unparseable JSON or almost entirely incorrect.

### Output format (strict)
Return a JSON object exactly in this format:
```json
{
"score": <integer 1-100>,
"summary": "<1-3 sentences explaining the main mismatches and strengths>"
}
```

Now evaluate:

GROUND_TRUTH_JSON
{answer}

ASSISTANT_JSON
{prediction}
"""

n_samples = int(os.getenv("TEXT2JSON_N_SAMPLES", -1))
text2json_judge_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role="HUMAN", prompt=GRADER_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=Text2JSONDataset,
            n_samples=n_samples,
            path="text2json",
            reader_cfg=text2json_reader_cfg,
        ),
        judge_cfg=dict(
            type=HuggingFacewithChatTemplate,
            path="Qwen/Qwen3-32B",
            abbr="qwen3-32b-judge",  # Model abbreviation
            model_kwargs=dict(attn_implementation="flash_attention_2"),
            chat_template_kwargs=dict(enable_thinking=False),
            # generation_kwargs=dict(
            #     max_new_tokens=4096
            # ),  # Maximum number of generated tokens
            max_out_len=1024,
            max_seq_len=132 * 1024,
            batch_size=1,  # Batch size
            run_cfg=dict(num_gpus=1),  # The required GPU numbers for this model
        ),
        dict_postprocessor=dict(type=text2json_llm_eval_postprocess),
    )
)

no_order = os.getenv("NO_ORDER", True)
no_order = bool(no_order)
if no_order:
    evaluator_type = Text2jsonEvaluator
else:
    evaluator_type = Text2jsonOrderedEvaluator

text2json_no_judge_eval_cfg = dict(
    evaluator=dict(type=evaluator_type),
    pred_role='BOT',
)

text2json_datasets = [
    dict(
        abbr="text2json",
        type=Text2JSONDataset,
        path="text2json",
        n_samples=n_samples,
        reader_cfg=text2json_reader_cfg,
        infer_cfg=text2json_infer_cfg,
        eval_cfg=text2json_no_judge_eval_cfg, # text2json_judge_eval_cfg,
    )
]
