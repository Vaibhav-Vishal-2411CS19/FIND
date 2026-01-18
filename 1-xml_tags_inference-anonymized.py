"""VQA Inference with vLLM - for MCQ type questions (Dataset V2)-
Multilingual - formatted output: reasoning and answer in xml-style tags"""

import argparse
import os

# Set global variables, environment variables, and logging configuration
# Environment setup
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0" # V100 - 7.0, T4,RTX 20xx -7.5, A100 - 8.0, RTX 30xx - 8.6, H100 - 9.0

## # START
os.environ["HF_HOME"] = "PLACEHOLDER_FOR_DIR_HF_HOME"  # Replace with actual path
os.environ["VLLM_CACHE_ROOT"] = "PLACEHOLDER_FOR_DIR_VLLM_CACHE" # Replace with actual path

print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

# _init_ function to setup logging
def _exp_init_ ():
    global BASE_EXP_DATA_SAVE_FOLDER
    BASE_EXP_DATA_SAVE_FOLDER = f"./Experiments-infer_xml-{datetime.now(tz=ZoneInfo('Asia/Kolkata')).strftime('%d_%b_%Y')}"

    os.makedirs(BASE_EXP_DATA_SAVE_FOLDER, exist_ok=True)

    global EXP_START_TIME
    EXP_START_TIME = datetime.now(tz=ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d_%H%M%S')
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s:: %(message)s",
        datefmt="%m-%d %H:%M:%S",
        filename=f"{BASE_EXP_DATA_SAVE_FOLDER}/inference_xml_format{EXP_START_TIME}.log",
        force=True
    )
    
    global logger   
    logger = logging.getLogger(__name__)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s %(asctime)s:: %(message)s", datefmt="%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)


# Standard library
import copy
import contextlib
import gc
import logging

import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from time import sleep

# Third-party libraries
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from vllm import LLM, SamplingParams

SUPPORTED_MODELS = {
    ### finetuned models
    "qwen5": "unsloth/Qwen3-VL-4B-Instruct",
    "qwen6": "unsloth/Qwen3-VL-8B-Instruct",
    
    "qwen1": "unsloth/Qwen2.5-VL-3B-Instruct",
    "qwen2": "unsloth/Qwen2.5-VL-7B-Instruct",
    
    "gemma1": "unsloth/gemma-3-4b-it",
    "gemma2": "unsloth/gemma-3-12b-it",

    "qwen7": "unsloth/Qwen3-VL-32B-Instruct",
    "qwen3": "unsloth/Qwen2.5-VL-32B-Instruct",
    "gemma3": "unsloth/gemma-3-27b-it",

    ### baseline models
    "gemma1": "google/gemma-3-4b-it",
    "gemma2": "google/gemma-3-12b-it",
    "gemma3": "google/gemma-3-27b-it",

    "qwen1": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3": "Qwen/Qwen2.5-VL-32B-Instruct",
    
    "qwen5": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen6": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen7": "Qwen/Qwen3-VL-32B-Instruct",
}

IMG_LLM = True
SEED = 42
MAX_IMG_DIM = 600
NUM_CPU = 1

LANGUAGES = ['english', 'hindi', 'bengali', 'marathi', 'gujarati', 'tamil']
LANG_CHAR_MAP = {"english": "ABCD", "hindi": "कखगघ", "bengali": "কখগঘ", "marathi": "कखगघ", "gujarati": "કખગઘ", "tamil": "௧௨௩௪"}

def cleanup_memory():
    """Clean up GPU memory and garbage collection."""
    torch.cuda.empty_cache()
    gc.collect()

def cleanup_vllm(llm: LLM):
    """Clean up vLLM resources."""
    if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'engine_core'):
        llm.llm_engine.engine_core.shutdown()

    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()

    cleanup_memory()


def process_image(example):
    example["image"] = example["image"].convert('RGB')
    if max(example["image"].size) > MAX_IMG_DIM:
        example["image"].thumbnail((MAX_IMG_DIM, MAX_IMG_DIM))
    return example

# For MCQ type dataset
def create_options_column(example, lang: str):
    example[f"Options-{lang}"] = (
        f"[{LANG_CHAR_MAP.get(lang, 'ABCD')[0]}] {example['Option a']} "
        f"[{LANG_CHAR_MAP.get(lang, 'ABCD')[1]}] {example['Option b']} "
        f"[{LANG_CHAR_MAP.get(lang, 'ABCD')[2]}] {example['Option c']} "
        f"[{LANG_CHAR_MAP.get(lang, 'ABCD')[3]}] {example['Option d']} "
    )
    return example

def load_and_process_dataset() -> object:
    """Load and preprocess the dataset."""
    logger.info("Loading dataset...")
    DATASET_NAME = "PLACEHOLDER_FOR_DATASET"  # Replace with actual dataset name
    
    # New MCQ Dataset columns: ['image', 'Question', 'Final Answer', 'Reasoning', 'Option a', 'Option b', 'Option c', 'Option d', 'Language', 'Question Level']
    dataset = load_dataset(DATASET_NAME, split="train")

    dataset = dataset.add_column("language_encoded", dataset["Language"])
    dataset = dataset.class_encode_column("language_encoded")

    dataset_split = dataset.train_test_split(test_size=0.3, seed=SEED, stratify_by_column="language_encoded") # 30% for testing
    dataset = dataset_split['test']

    # rename column 'Final Answer' to 'Answer'
    dataset = dataset.rename_column("Final Answer", "Answer")
    
    dataset = dataset.filter(lambda row: row['Answer'] is not None, num_proc=NUM_CPU)
    dataset = dataset.flatten_indices()

    if "images" in dataset.column_names:
        dataset = dataset.rename_column("images", "image")      
        dataset = dataset.filter(lambda row: row['image'] is not None, num_proc=NUM_CPU)
        dataset = dataset.map(process_image, num_proc=NUM_CPU)

    for lang in LANGUAGES:
        dataset = dataset.map(create_options_column, fn_kwargs={"lang": lang}, num_proc=NUM_CPU)
    logger.info(f"Dataset columns after pre-processing: {dataset.column_names}")

    time.sleep(1)
    
    global IMG_LLM
    if "image" not in dataset.column_names:
        IMG_LLM = False
        logger.warning("No 'image' column found in dataset. Proceeding as text-only.")
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    return dataset


def create_base_messages(lang: str) -> list[dict]:
    """Create base message structure with few-shot examples and prompt variations."""
    curr_lang_letters = LANG_CHAR_MAP.get(lang, 'ABCD')
    base_messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"""You are a helpful assistant. You will be given a multiple-choice question (MCQ). Solve the question \
and choose the correct option from the given choices ({curr_lang_letters[0]}, {curr_lang_letters[1]}, {curr_lang_letters[2]}, or {curr_lang_letters[3]}). Explain your reasoning step by step before providing your final answer.

Format your output as:
<reasoning> reasoning_text </reasoning>
<answer> Option [{curr_lang_letters[0]}] or [{curr_lang_letters[1]}] or [{curr_lang_letters[2]}] or [{curr_lang_letters[3]}] </answer>

The question and the options are given in {lang} language. Provide the answer and reasoning in the same language - {lang}, and the tags in English as shown above.
The question and the possible answers are as follows:"""}
        ],
    }]

    return base_messages

def create_prompt(messages: list[dict], image: object, question: str, options: str, processor: AutoProcessor) -> str:
    """Create VQA prompt with image and question."""
    temp_message = copy.deepcopy(messages)
    
    if IMG_LLM and (image is not None):
        temp_message[-1]["content"].append({"type": "image", "image": image})
    
    temp_message[-1]["content"].append({"type": "text", "text": f"\nQuestion: {question}"})
    temp_message[-1]["content"].append({"type": "text", "text": f"\nOptions: {options}"})

    prompt = processor.apply_chat_template(temp_message, tokenize=False, add_generation_prompt=True)
    return prompt

def prepare_prompts(dataset: any, processor: AutoProcessor, lang: str) -> tuple[list, dict[str, list[str]], list[str]]:
    """Prepare all prompts for inference."""
    base_messages = create_base_messages(lang)

    prompts = []
    questions = dataset['Question']

    logger.info("Preparing prompts...")
    for item in tqdm(dataset):
        if IMG_LLM == True:
            prompt = create_prompt(base_messages, item['image'], item['Question'], item[f'Options-{lang}'], processor)
            prompts.append({"prompt": prompt,
                            "multi_modal_data": {"image": [item['image']]}
                            })
        else:
            prompt = create_prompt(base_messages, None, item['Question'], item[f'Options-{lang}'], processor)
            prompts.append({"prompt": prompt})
    
    ground_truths = {
        'Answer': dataset['Answer'],
        'Reasoning': dataset['Reasoning'],
        'Answer_and_Reason': [f"{ans}{reason}" for ans, reason in zip(dataset['Answer'], dataset['Reasoning'])]
    }
    return prompts, ground_truths, questions


def run_inference(llm: LLM, prompts: list, sampling_params: SamplingParams) -> list[str]:
    """Run inference on all prompts."""
    logger.info("Running inference...")

    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def parse_predictions(predictions: list[str]) -> dict[str, list[str]]:
    """Parse predictions to extract Answer and Reasoning."""
    answer_pattern = re.compile(
    r"""
    <answer>\s*(?P<answer>.*?)\s*</answer>              # Capture content between answer tags
    .*?                                                 # Match any noise/whitespace between tags
    (?:<reasoning>\s*(?P<reasoning>.*?)\s*</reasoning>)? # Capture content between reasoning tags (Optional)
    """,
    re.DOTALL | re.VERBOSE | re.IGNORECASE)

    predictions_answer = []
    predictions_reasoning = []

    for pred in predictions:
        match = answer_pattern.search(pred)
        if match:
            predictions_answer.append(match.group("answer").strip() if match.group("answer") else "")
            predictions_reasoning.append(match.group("reasoning").strip() if match.group("reasoning") else "")
        else:
            predictions_answer.append("")
            predictions_reasoning.append("")

    return {
        "Answer": predictions_answer,
        "Reasoning": predictions_reasoning,
        "Answer_and_Reason": predictions
    }

def evaluate_results(predictions: list[str], ground_truths: list[str], lang: str) -> dict[str, float]:
    """Evaluate predictions"""
    results = {}

    curr_lang_chars = LANG_CHAR_MAP.get(lang, "ABCD")
    pred_options = [
        (m.group(0) if m else "N/A") for m in (re.search(rf"[{curr_lang_chars}]", str(text).upper()) for text in predictions)]
    gt_options = [gt.upper() for gt in ground_truths]
    correct = [1 if g == p else 0 for g, p in zip(gt_options, pred_options)]
    accuracy = sum(correct) / len(correct)
    results['accuracy'] = accuracy
    return results

def save_results(
    predictions: dict[str, list[str]],
    ground_truths: dict[str, list[str]],
    questions: list[str],
    model_id: str,
    lang: str
):
    """Save results and metrics to files."""
    model_name = model_id.split('/')[-1]
    model_name = model_name.replace('/', '_')
    filename_base = f"{BASE_EXP_DATA_SAVE_FOLDER}/{model_name}_{lang}"

    # Save detailed results
    results_df = pd.DataFrame({
        'question': questions,
        'answer_ground_truth': ground_truths['Answer'],
        'answer_prediction': predictions['Answer'],
        'reasoning_ground_truth': ground_truths['Reasoning'],
        'reasoning_prediction': predictions['Reasoning'],
        'both_ground_truth': ground_truths['Answer_and_Reason'],
        'both_prediction': predictions['Answer_and_Reason'],
    })
    RESULTS_FILE_NAME = f'{filename_base}_{EXP_START_TIME}_results.csv' 
    results_df.to_csv(RESULTS_FILE_NAME, index=False)

    # Evaluate and save metrics
    result = evaluate_results(predictions['Answer'], ground_truths['Answer'], lang)
    
    result.update({
        'model': model_id,
        'language': lang,
        'output': 'xml_style_tags',
        'total_samples': len(predictions['Answer'])
    })

    metrics_df = pd.DataFrame([result])
    METRICS_FILE_NAME = f'{filename_base}_{EXP_START_TIME}_metrics.csv'
    metrics_df.to_csv(METRICS_FILE_NAME, index=False)

    logger.info(f"Results saved: {RESULTS_FILE_NAME}, {METRICS_FILE_NAME}")
    logger.info(f"Accuracy: {result['accuracy']:.4f}")


def find_nested_key(data, target_key):
    """
    Recursively search for a key in a nested dictionary or list.
    Returns the first occurrence of the key's value or None if not found.
    """
    if isinstance(data, dict):
        # Check if the target key exists in the current dictionary
        if target_key in data:
            return data[target_key]
        # Recurse through all values in the dictionary
        for value in data.values():
            result = find_nested_key(value, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        # Recurse through all items in the list
        for item in data:
            result = find_nested_key(item, target_key)
            if result is not None:
                return result
    return None

def load_model(model_name: str) -> tuple[LLM, AutoProcessor, SamplingParams]:
    """Load model, processor, and sampling parameters."""
    logger.info(f"\n\nLoading model: {model_name}\n")
    
    # Conditionally build the arguments dictionary
    args_to_pass = {}
  
    # Load the config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True).to_dict()

    # Check max position embeddings
    max_pos = find_nested_key(config, "max_position_embeddings")
    logger.info(f"HF Config max_position_embeddings: {max_pos}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    
    # if IMG_LLM is unset, then check if the model supports images
    if IMG_LLM is not None:    
        logger.info(f"IMG_LLM is set to {IMG_LLM} externally.")

    if max_pos is not None:
        args_to_pass['max_model_len'] = min(1024*10, max_pos)
        # args_to_pass['max_seq_len_to_capture'] = min(1024*24, max_pos) # Depreciated in vLLM V1 engine

    if IMG_LLM == True:
        args_to_pass['limit_mm_per_prompt'] = {"image": 1} 
    
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        task="generate",
        tensor_parallel_size=TENSOR_PARALLEL_COUNT,
        pipeline_parallel_size=PIPELINE_PARALLEL_COUNT,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        # max_parallel_loading_workers=1,
        enforce_eager=True, # fast debug runs
        **args_to_pass
    )


    sampling_params = llm.get_default_sampling_params()
    logger.info(f"Sampling Params used -{sampling_params}")

    return llm, processor, sampling_params


def run_single_experiment(
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    dataset: object,
    model_name: str,
    lang: str
):
    """Run a single experiment configuration."""

    # Check if experiment already completed
    model_name = model_name.replace('/', '_')
    filename_base = f"{BASE_EXP_DATA_SAVE_FOLDER}/{model_name}_{lang}"
    results_file = f'{filename_base}_results.csv'
    metrics_file = f'{filename_base}_metrics.csv'

    if os.path.exists(results_file) and os.path.exists(metrics_file):
        logger.info(f"Skipping experiment: {lang} language (files already exist)")
        return

    # filter dataset for the current language
    dataset = dataset.filter(lambda example: example['Language'] == lang)

    start_time = time.perf_counter()

    # Prepare prompts
    prompts, ground_truths, questions = prepare_prompts(dataset, processor, lang)
    logger.info(f"Sample prompt-\n{prompts[0]}")

    # Run inference
    predictions = run_inference(llm, prompts, sampling_params)
    # Parse predictions
    processed_predictions = parse_predictions(predictions)

    # Save results
    save_results(processed_predictions, ground_truths, questions, model_name, lang)

    # Clean up prompts from memory
    del prompts, ground_truths, questions, predictions, processed_predictions
    cleanup_memory()

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Experiment completed in {elapsed_time:.2f} seconds")

def model_path_sft(model_name: str) -> str:
    FT_LLM_DIR = "PLACEHOLDER_FOR_DIR_FINETUNED_MODELS"  # Replace with actual path
    SFT_MODEL_PATH = os.path.join(FT_LLM_DIR, model_name.replace("/", "-") + "-sft-EXP_ID") # Replace EXP_ID with actual experiment ID, eg. text-ep03
    return SFT_MODEL_PATH

def main():
    """Main execution function."""
    logger.info("Starting VQA inference experiments")

    # Load dataset and few-shot examples
    dataset = load_and_process_dataset()

    # Run all experiment combinations
    total_experiments = len(SUPPORTED_MODELS.values()) * len(LANGUAGES)
    experiment_count = 0

    for experiment_count, (model_id, model_name) in enumerate(SUPPORTED_MODELS.items()):
        if 'unsloth' in model_name.lower():
            # loading sft'ed models
            model_name = model_path_sft(model_name)
        global MODEL_ID
        MODEL_ID = model_name
        logger.info(f"Evaluating Model: {model_name}\n")
        sleep(5)  # Brief pause before loading new model
        try: 
            llm, processor, sampling_params = load_model(model_name)
            
            for lang in LANGUAGES:
                logger.info(f"Experiment {experiment_count}/{total_experiments} - \n Model: {model_name},"
                            f"Language {lang}")
                run_single_experiment(llm, processor, sampling_params, dataset, model_name, lang)

            cleanup_vllm(llm)
            sleep(10)  # Allow time for resources to be released
        except Exception as e:
            logger.exception(f"Error occurred while evaluating model {model_name}: {e}")
            cleanup_memory()
            continue
    logger.info("All experiments completed successfully")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_idx", type=str, default="qwen1", help="Index of the model to run")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism count")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism count")
    args = parser.parse_args()
    

    global RAG_EXPERIMENT, MODEL_IDX, TENSOR_PARALLEL_COUNT, PIPELINE_PARALLEL_COUNT
    RAG_EXPERIMENT = False
    MODEL_IDX = args.model_idx
    TENSOR_PARALLEL_COUNT = args.tp
    PIPELINE_PARALLEL_COUNT = args.pp
    _exp_init_()

    main()
    