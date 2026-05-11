from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.text2json import LongProcDataset, LongProcEvaluator


DATASET_NAME = "html_to_tsv_8k"

longproc_reader_cfg = dict(input_columns=["problem"], output_column="reference_output")
longproc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="{input_prompt}",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024), 
)

longproc_eval_cfg = dict(
    evaluator=dict(type=LongProcEvaluator, path="longproc", dataset_name=DATASET_NAME),
    pred_role='BOT',
)

longproc_dataset = [
    dict(
        abbr="longproc",
        type=LongProcDataset,
        path="longproc",
        dataset_name=DATASET_NAME,
        reader_cfg=longproc_reader_cfg,
        infer_cfg=longproc_infer_cfg,
        eval_cfg=longproc_eval_cfg,
    )
]
