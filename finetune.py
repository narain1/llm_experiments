from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
    BitsAndBytesConfig,
    GenerationConfig
)

import time
import random
from dataclasses import dataclass, field
from typing import Optional
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    model_path: str
    model_dtype: str
    use_lora: bool
    lora_alpha: float
    lora_r: int
    lora_modules: str
    max_seq_length: int
    answer_delimiter: str
    train_file: str
    val_file: str
    num_proc: int
    num_context: int
    prompt_template: str


def collate(examples, tokenizer, max_length, template):
    prompts = [e['prompt'] for e in examples]
    contexts = [e['context'] for e in examples]

    answer_texts, wrong_answer_text = [], []

    for e in examples:
        answer_letter = e["answer"]
        answer_texts.append(e[answer_letter])
        wrong_answers = [e[letter] for letter in "ABCDE" if letter != answer_letter]
        wrong_answer_texts.append(wrong_answers)

    full_prompts = []
    correct_tokens = []

    for context, prompt, answer_text, wrong_answers in zip(
        contexts, prompts, answer_texts, wrong_answer_texts
    ):
        options = [answer_text] + wrong_answers
        random.shuffle(options)

        answer_delimiters = "ABCDE"
        correct_tokens.append(answer_delimiters[options.index(answer_text)])
        full_prompts.append(
            template.format(
                context=context,
                prompt=prompt,
                A=options[0],
                B=options[1],
                C=options[2],
                D=options[3],
                E=options[4],
                answer=answer_delimiters[options.index(answer_text)],
            )
        )

    inputs = tokenizer(
        full_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=16,
    )

    labels = inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        **inputs,
        "labels": labels,
    }



