#!/usr/bin/env python3
"""Train a Qwen LoRA adapter for tactical scene reasoning.

This is the reliable H100 path: Gemini labels multimodal scene packs, then this
script trains a local model over the distilled structured evidence and answers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False

    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": 0.05,
        "logging_steps": 1,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "packing": False,
        "report_to": "none",
    }
    sft_fields = getattr(SFTConfig, "__dataclass_fields__", {})
    if "max_length" in sft_fields:
        sft_kwargs["max_length"] = args.max_seq_length
    else:
        sft_kwargs["max_seq_length"] = args.max_seq_length
    training_args = SFTConfig(**sft_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "formatting_func": lambda ex: format_example(tokenizer, ex),
    }
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "base_model": args.model,
        "train_jsonl": args.train_jsonl,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "load_in_4bit": args.load_in_4bit,
    }
    Path(args.output_dir, "narya_training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"saved adapter to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", default="ml/data/sft_train.jsonl")
    parser.add_argument("--output-dir", default="ml/runs/qwen-tactical-lora")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def format_example(tokenizer: Any, example: dict[str, Any]) -> str:
    messages = example["messages"]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        parts = []
        for msg in messages:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        return "\n\n".join(parts) + tokenizer.eos_token


if __name__ == "__main__":
    main()
