import os
os.environ["HF_HOME"] = "PLACEHOLDER_FOR_DIR_HF_HOME"  # Replace with actual path
os.environ["HF_HUB_OFFLINE"] = "1"

CACHE_PATH = "PLACEHOLDER_FOR_DIR_CACHE" # Set UnsLoTH cache path
os.makedirs(CACHE_PATH, exist_ok=True)
os.environ["UNSLOTH_CACHE"] = CACHE_PATH
os.environ["TMPDIR"] = CACHE_PATH # ALSO set TMPDIR inside python just to be safe regarding the lock file

from pathlib import Path

from unsloth import FastVisionModel # FastLanguageModel for VLMs
import torch

import gc
import time


def load_model(MODEL_NAME: str):
    def get_hf_snapshot_path(model_name: str, hf_home: str) -> str:
        """
        Resolve the latest snapshot path for a HF cached model.
        """
        org, repo = model_name.split("/")
        model_dir = Path(hf_home) / "hub" / f"models--{org}--{repo}" / "snapshots"

        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found in HF cache: {model_dir}")

        snapshots = sorted(
            [p for p in model_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not snapshots:
            raise FileNotFoundError(f"No snapshots found in {model_dir}")

        return str(snapshots[0])
    
    
    HF_HOME = os.environ["HF_HOME"]
    LOCAL_MODEL_PATH = get_hf_snapshot_path(MODEL_NAME, HF_HOME)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = LOCAL_MODEL_PATH,
        max_seq_length = 1024 * 4,   # Context length - can be longer, but uses more memory
        load_in_4bit = False,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
        local_files_only = True,
    )


    model = FastVisionModel.get_peft_model(
        model,
        r = 8,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj"],
        # , "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0.1, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )
    return model, tokenizer

def datatset_prep(tokenizer):
        
    from datasets import load_dataset
    DATASET_NAME = "PLACEHOLDER_FOR_DATASET"  # Replace with actual dataset name
    dataset = load_dataset(DATASET_NAME, split = "train")

    LANG_CHAR_MAP = {"english": "ABCD", "hindi": "कखगघ", "bengali": "কখগঘ", "marathi": "कखगघ", "gujarati": "કખગઘ", "tamil": "௧௨௩௪"}
    def create_options_column(example):
        lang = example['Language']
        example[f"Options"] = (
            f"[{LANG_CHAR_MAP.get(lang, 'ABCD')[0]}] {example['Option a']} "
            f"[{LANG_CHAR_MAP.get(lang, 'ABCD')[1]}] {example['Option b']} "
            f"[{LANG_CHAR_MAP.get(lang, 'ABCD')[2]}] {example['Option c']} "
            f"[{LANG_CHAR_MAP.get(lang, 'ABCD')[3]}] {example['Option d']} "
        )
        return example

    MAX_IMG_DIM = 512
    def process_image(example):
        example["image"] = example["image"].convert('RGB')
        if max(example["image"].size) > MAX_IMG_DIM:
            example["image"].thumbnail((MAX_IMG_DIM, MAX_IMG_DIM))
        return example

    dataset = dataset.map(create_options_column, num_proc=1)

    if 'image' in dataset.column_names:
        dataset = dataset.map(process_image, num_proc=2)

    
    def convert_to_conversation(sample):
        # format the dataset for training
        lang = sample['Language']
        curr_lang_letters = LANG_CHAR_MAP.get(lang, 'ABCD')
        instruction = f"""You are a helpful assistant. You will be given a multiple-choice question (MCQ). Solve the question \
and choose the correct option from the given choices ({curr_lang_letters[0]}, {curr_lang_letters[1]}, {curr_lang_letters[2]}, or {curr_lang_letters[3]}). 

The output should be a single letter only corresponding to the correct option:
{curr_lang_letters[0]} or {curr_lang_letters[1]} or {curr_lang_letters[2]} or {curr_lang_letters[3]}
Do not output anything else.
The question and the options are given in {lang} language. Provide the answer option in the same language - {lang}
The question and possible answers are as follows:"""

        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : instruction},
                {"type" : "text",  "text"  : f"Question: {sample['Question']} "},
                {"type" : "text", "text" : "Options: " + sample[f'Options']},]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : f"{sample['Final Answer']}" }] 
            },
        ]
        return conversation

    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    reasoning_conversations = tokenizer.apply_chat_template(
        converted_dataset,
        tokenize = False,
    )

    print(f'\n{'*'*10}\n',reasoning_conversations[0], f'\n{'*'*10}\n')


    import pandas as pd
    dataset_conversation = pd.DataFrame({"text":reasoning_conversations})

    from datasets import Dataset
    processed_dataset = Dataset.from_pandas(pd.DataFrame(dataset_conversation))

    processed_dataset = processed_dataset.add_column("language_encoded", dataset["Language"])
    processed_dataset = processed_dataset.class_encode_column("language_encoded")

    dataset_split = processed_dataset.train_test_split(test_size=0.3, seed=42, stratify_by_column="language_encoded") # 30% for testing
    train_dataset = dataset_split['train']
    test_dataset = dataset_split['test']

    print(f"Train dataset size: {len(train_dataset)}")
    print(f'\n{'*'*10}\n', test_dataset[0], f'\n{'*'*10}\n')
    
    return train_dataset, test_dataset

def train_model(model, tokenizer, train_dataset, test_dataset, model_name):
    from trl import SFTTrainer, SFTConfig
    
    FastVisionModel.for_training(model)

    output_dir_base = "PLACEHOLDER_FOR_DIR_OUTPUT"  # Replace with actual path
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 16, #16 for all exp, other than Qwen2.5,32B,ep03 Try to keep this as high as possible!
            gradient_accumulation_steps = 8, #8 for all exp, Use GA to mimic batch size!
            # Use warmup_ratio and num_train_epochs for longer runs!
            # max_steps = 10,
            # warmup_steps = 5,
            num_train_epochs = 3, # Set this for 1 full training run.
            warmup_ratio = 0.1,
            learning_rate = 1e-5, # Reduce to 2e-5 for long training runs
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            max_length=1024 * 10, # Max length of text input
            eval_strategy="steps",
            eval_steps=30, # 30 for exp, other than
            save_strategy="epoch",
            output_dir = output_dir_base + model_name.split('/')[1] + "-sft-img-ep01" + time.strftime("-%Y%m%d-%H%M%S"),
            # report_to = "wandb", # Use TrackIO/WandB etc
        ),
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"Peak reserved memory = {used_memory} GB., for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %, for training = {lora_percentage} %.")


model_list = [
    "unsloth/Qwen3-VL-4B-Instruct",
    "unsloth/Qwen3-VL-8B-Instruct",

    "unsloth/Qwen2.5-VL-3B-Instruct",
    "unsloth/Qwen2.5-VL-7B-Instruct",

    "unsloth/gemma-3-4b-it",
    "unsloth/gemma-3-12b-it",
    
    "unsloth/gemma-3-27b-it",
    "unsloth/Qwen3-VL-32B-Instruct",
    "unsloth/Qwen2.5-VL-32B-Instruct",
]

for MODEL_NAME in model_list:
    print(f"\n\n{'#'*20}\nStarting training for model: {MODEL_NAME}\n{'#'*20}\n\n")

    try:
        # 2. Load Model & Tokenizer
        model, tokenizer = load_model(MODEL_NAME)

        # 3. Format dataset specifically for this model/tokenizer        
        train_dataset, test_dataset = datatset_prep(tokenizer)

        # 4. Train
        train_model(model, tokenizer, train_dataset, test_dataset, MODEL_NAME)

        # 5. Save
        FT_LLM_DIR = "PLACEHOLDER_FOR_DIR_FINETUNED_MODELS"  # Replace with actual path
        SFT_MODEL_PATH = os.path.join(FT_LLM_DIR, MODEL_NAME.replace("/", "-") + "-sft-text-ep_03") # Change exp_id: img-ep01 as needed
        model.save_pretrained_merged(SFT_MODEL_PATH, tokenizer, save_method = "merged_16bit",)

    except Exception as e:
        print(f"FAILED on {MODEL_NAME}: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        SLEEP_TIME = 2
        time.sleep(SLEEP_TIME)
        # 6. Cleanup to prevent OOM on next loop
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        if 'trainer' in locals(): del trainer
        if 'train_dataset' in locals(): del train_dataset
        if 'test_dataset' in locals(): del test_dataset
        gc.collect()
        time.sleep(SLEEP_TIME)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(SLEEP_TIME)
