### Scripts to train the Mistral-7B-instruct v0.1 

# pypi packages 
import pickle
import os 
from transformers import (
    AutoTokenizer, 
    default_data_collator,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
)
import pandas as pd 
from datasets import Dataset 
from peft import (
    LoraConfig,
    get_peft_model,
)
import argparse

# initailize the environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# utility functions for chat model training 
def find_intervals_same(lst, element):
    intervals = []
    i = 0
    while i < len(lst):
        if lst[i] == element:
            start = i
            i += 1
            while i < len(lst) and lst[i] != element:
                i += 1
            if i < len(lst):
                end = i
                intervals.append((start, end))
        i += 1

    return intervals

def find_intervals(lst, start_seq, end_seq):

    intervals = []
    i = 0
    while i < len(lst) - len(start_seq) + 1:
        if lst[i:i+len(start_seq)] == start_seq:
            for j in range(i + len(start_seq), len(lst) - len(end_seq) + 1):
                if lst[j:j+len(end_seq)] == end_seq:
                    if j > i + len(start_seq):  
                        intervals.append((i + len(start_seq), j - 1))
                    i = j + len(end_seq) - 1
                    break
            else:
                i += len(start_seq)
                continue
        i += 1

    return intervals

def find_mask_start_end(inds, tokenizer, start_seq=None, end_seq = None, start_token_id = None, end_token_id = None , offset = 1, special = True):
    """
    No need for two pointer trick here -> after testing the running time 
    """
    if special: 
        start_id = tokenizer.encode(start_seq)[1:]
        end_id = tokenizer.encode(end_seq)[1:]

        start = []
        end = []
        intervals = find_intervals(inds, start_id, end_id)
        for i, int in enumerate(intervals):
            start.append(int[0])
            end.append(int[1]+1)
    else:
        start = []
        end = []
        intervals = find_intervals_same(inds, start_token_id)
        for i, int in enumerate(intervals):
            start +=  [int[0]+1 + offset]
            end +=  [int[1] - offset]

    return start, end

def preprocess_function(examples, tokenizer, text_column = "text", max_length = 1500):

    batch_size = len(examples[text_column])
    model_inputs = tokenizer(examples[text_column])
    labels = tokenizer(examples[text_column])

    for i in range(batch_size):  
        start_para, end_para = find_mask_start_end(labels["input_ids"][i], start_seq="Start of the retrieval text", end_seq="End of the retrieval text", special = True)
        if len(start_para) != 0:
            for start, end in zip(start_para, end_para):
                labels["input_ids"][i][start:end] = [-100] * (end - start)
        start_inst, end_inst = find_mask_start_end(labels["input_ids"][i], start_token_id=16289,  end_token_id=16289, special= False)
        if len(start_inst) != 0:
            for start, end in zip(start_inst, end_inst):
                labels["input_ids"][i][start:end] = [-100] * (end - start)
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
        model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][:max_length]
        labels["input_ids"][i] = labels["input_ids"][i][:max_length]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def parse_args():

    parser = argparse.ArgumentParser(description = "parser for training the Mistral-7B-instruct model")
    parser.add_argument("--model_name", type = str, default = "mistralai/Mistral-7B-Instruct-v0.1", help = "the model name for training")
    parser.add_argument("--train_file", type = str, help = "the training file for the model")
    parser.add_argument("--lora_rank", type = int, default = 8, help = "Lora rank for the model")
    parser.add_argument("--lora_alpha", type = int, default = 32, help = "Lora alpha for the model")
    parser.add_argument("--lora_dropout", type = float, default = 0.05, help = "Lora dropout for the model")
    parser.add_argument("--num_train_epochs", type = int, default = 2, help = "the number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type = int, default = 2, help = "the per device batch size for training")
    parser.add_argument("--gradient_accmulation_steps", type = int, default = 2, help = "the gradient accumulation steps")
    parser.add_argument("--learning_rate", type = float, default = 5e-5, help = "the learning rate for the model")
    parser.add_argument("--output_dir", type = str, help = "the output directory for the model")

    args = parser.parse_args()

    return args 

def main():

    args = parse_args() 
    
    with open(args.train_file, "rb") as f:
        train_data = pickle.load(f)
    
    ds_train = Dataset.from_pandas(pd.DataFrame(train_data).rename(columns = {0: "text"}), split = "train")
    ds_train = ds_train.map(
        preprocess_function,
        batched = True,
        num_proc = 1,
        remove_columns = ds_train.column_names,
        load_from_cache_file = False,
        desc = "Running Tokenizer on the dataset",
    )
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 low_cpu_mem_usage = True,
                                                 use_cache = False,
                                                 device_map = "auto",
                                                 )
    config = LoraConfig(
        r = args.lora_rank,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        bias = "none",
        target_modules= ["q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "k_proj", "up_proj"], 
        task_type = "CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    print(model.print_trainable_parameters())

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        num_train_epochs = args.num_train_epochs,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accmulation_steps,
        weight_decay = 1e-4,
        dataloader_drop_last = True,
        bf16= True,
        logging_steps = 10, 
        learning_rate = args.learning_rate,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        remove_unused_columns = False,
        dataloader_num_workers = 4,
        dataloader_prefetch_factor = 2,
        lr_scheduler_type = "cosine",
        )

    trainer = Trainer(
        model = model, 
        args = training_args, 
        train_dataset = ds_train, 
        data_collator = default_data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    
    main()