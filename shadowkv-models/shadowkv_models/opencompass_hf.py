################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

from transformers import PreTrainedTokenizerBase

from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate


def _ensure_batch_encode_plus(tokenizer):
    if hasattr(tokenizer, "batch_encode_plus"):
        return
    tokenizer.batch_encode_plus = lambda batch_text_or_text_pairs, **kwargs: tokenizer(
        batch_text_or_text_pairs, **kwargs
    )


if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
    PreTrainedTokenizerBase.batch_encode_plus = (
        lambda self, batch_text_or_text_pairs, **kwargs: self(batch_text_or_text_pairs, **kwargs)
    )


class HuggingFacewithChatTemplatePatched(HuggingFacewithChatTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _ensure_batch_encode_plus(self.tokenizer)
