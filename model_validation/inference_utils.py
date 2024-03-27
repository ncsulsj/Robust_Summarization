### This file along with llm_utils.py contains the utility functions for the model validation. 
### pypi packages
from typing import Optional, List, Tuple 
from huggingface_hub import snapshot_download
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
import openai 
from dotenv import load_dotenv
from evaluate import load 
import numpy as np 
import os 
import torch 
import re 
import logging

# initalize environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
torch.set_grad_enabled(False)
logger = logging.getLogger(__name__)

# utility functions (The following split_input_label functions are to extract the user topic/own'text and the retrieval text from the test data to evaluate other model's performance)
def split_input_and_label_1_2_4_5_6(input_text):

    inp_match = re.search(r'(.*?End of the retrieval text)', input_text, re.DOTALL)
    inp = inp_match.group(1) if inp_match else ""
    label_match = re.search(r'End of the retrieval text\.(.*?)</s>', input_text, re.DOTALL)
    label = label_match.group(1) if label_match else ""

    return inp, label 

def split_input_and_label_3(input_text):

    pattern = r"(.*?)There is no need to retrieve text since user provides own text."
    inp_match = re.search(pattern, input_text, re.S)
    inp = inp_match.group(1).strip() if inp_match else ""
    pattern = r"There is no need to retrieve text since user provides own text\.(.*?)</s>"
    label_match = re.findall(pattern, input_text, re.DOTALL)
    label = label_match[0] if len(label_match)>0 else ""

    return inp, label 

def split_input_and_label_7(input_text):

    if "The final summarization is" not in input_text:
        end_index = input_text.find("End of the retrieval text", input_text.find("End of the retrieval text") + 1)
        end_index_s =  input_text.find("</s>", input_text.find("</s>") + 1)
        inp = input_text[:end_index] + " End of the retrieval text. " if end_index != -1 else ""
        label = input_text[end_index: end_index_s].replace("End of the retrieval text. ", "")
    else:
        pattern_before = r"(.*?\[/INST\])"
        pattern_after = r"\[/INST\](.*)"
        match_before = re.search(pattern_before, input_text)
        match_after = re.search(pattern_after,input_text)
        if match_before:
            inp = match_before.group(1)
        else:
            logger.error("No [/INST] tag found")
        if match_after:
            label = match_after.group(1)
        else:
            logger.error("No [/INST] tag found") 

    return inp, label


def split_input_and_label(input_text, scenairo_type: int):

    match scenairo_type:
        case 1: 
            inp, label = split_input_and_label_1_2_4_5_6(input_text)
        case 2:
            inp, label = split_input_and_label_1_2_4_5_6(input_text)
        case 3:
            inp, label = split_input_and_label_3(input_text)
        case 4: 
            inp, label = split_input_and_label_1_2_4_5_6(input_text)
        case 5: 
            inp, label = split_input_and_label_1_2_4_5_6(input_text)
        case 6: 
            inp, label = split_input_and_label_1_2_4_5_6(input_text)
        case 7: 
            if "[INST] You are a summarization assistant to summarize the documents one by one" in input_text:
                inp, label = split_input_and_label_1_2_4_5_6(input_text)
            else:
                inp, label = split_input_and_label_7(input_text)
    
    return inp, label 
                

def create_test_prompt(
        lora_path: str, 
        text_list: List[str],
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    
    prompts = [] 
    for i, text in enumerate(text_list):
        prompts.append(
            (
                text, 
                SamplingParams(temperature = 0.1, 
                               max_tokens = 1500, 
                               stop = ["</s>"]),
                LoRARequest("scenairo", 1, lora_path), 
            )
        )

    return prompts 

def process_requests(engine: LLMEngine, 
                     test_prompts:  List[Tuple[str, SamplingParams, Optional[LoRARequest]]],
                     use_lora):
    
    request_id = 0 
    results = []
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts: 
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            if use_lora:
                engine.add_request(str(request_id), 
                                prompt, 
                                sampling_params, 
                                lora_request = lora_request)
            else:
                engine.add_request(str(request_id), 
                                prompt, 
                                sampling_params, 
                                )
            request_id += 1 
        request_outputs: List[RequestOutput] = engine.step() 
        for request_output in request_outputs:
            if request_output.finished:
                results.append({"pred": request_output.outputs[0].text, "idx": request_output.request_id})

    return results

def initialize_engine() -> LLMEngine:

    engine_args = EngineArgs(model = "mistralai/Mistral-7B-Instruct-v0.1", 
                             enable_lora = True, 
                             max_loras = 1, 
                             max_lora_rank = 32, 
                             max_cpu_loras = 2, 
                             max_num_seqs = 256)
    
    return LLMEngine.from_engine_args(engine_args)


def check_logic(scenairo, label, prediction):

    if scenairo == 1:
        logic = "The prediction text should catch the irrelevancy of the retrieval text with respect to user's query"
    elif scenairo == 2:
        logic = "The prediction text should catch the relevancy of the retrieval text with respect to user's query"
    elif scenairo == 3:
        logic = "The prediction text should state that there is no need to retrieve the text."
    elif scenairo == 4:
        logic = "The prediction text should catch the irrelevancy of the retrieval text with respect to user's text"
    elif scenairo == 5:
        logic = "The prediction text should catch the relevancy of the retrieval text and think the retrieval text should combine with user's text to do summarization"
    elif scenairo == 6:
        logic = "The prediction text should catch the relevancy of the retrieval text and think the retrieval text has information conflict with the user's text"
    elif scenairo == 7:
        logic = """There are three things to check. First, the prediction text should have the relevancy of the retrieval text to be same as the true text; second, 
        the context information inside the prediction text should be the same as the one in true text; finally, based on the relevancy of the retrieval text, the prediction text
        should have the same decision to combine the retrieval text or not when doing final summarization.
        """
    else:
        raise ValueError("no such case exist =======>")

    if scenairo == 7:
        prompt =f"""
        Here is the prediction text: {prediction}\n 
        Here is the true text: {label}\n 
        
        You need to follow following criteria ###{logic}###to check the prediction text. If the criteria meets, return me a single number 1; if not, return me a single number 0; 
        (You do not need to tell how you arrive the result, just give me a single number!)
        """
    else:
        prompt = f"""
        Here is the prediction text: {prediction}\n
        Here is the logic you need to check following criteria ###{logic}###. If the criteria meets, return me a single number 1; if not, return me a single number 0; 
        (You do not need to tell how you arrive the result, just give me a single number!)
        """
        

    response = openai.ChatCompletion.create(
        model = "gpt-4-1106-preview",
        messages = [{"role": "system", "content": "You are a helpful assistant to follow the instruction"},
                    {"role": "user", "content": prompt}],
        temperature = 0,
    )

    return response["choices"][0]["message"]["content"]


def check_coherence_quality(metric, labels, predictions):

    labels = [pick_up_summary(label) for label in labels]
    predictions = [pick_up_summary(prediction) for prediction in predictions]

    if metric == "bertscore": 
        bertscore = load("bertscore")
        results = bertscore.compute(predictions = predictions, references = labels, lang = "en")
        return {
            "precision": (np.mean(results["precision"]), np.std(results["precision"])),
            "recall": (np.mean(results["recall"]), np.std(results["recall"])),  
            "f1": (np.mean(results["f1"]), np.std(results["f1"]))
        }
    elif metric == "rouge": 
        rouge = load("rouge")
        results = rouge.compute(predictions=predictions, references=labels)

        return results 
    
    else: 

        raise ValueError("The metric is not supported yet. ") 
    
def pick_up_summary(text):

    prompt = f"""
    Here is the text provided to you: ###
    {text} ### \n 
    This is a text which consists of how an assistant to derive the summarization. Ignore the derivation process, and copy the summarization
    inside the text and return it to me. (The response you return to me should be exactly the summarization, do not give me extra information)
    """
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0125",
        messages = [{"role": "system", "content": "You are a helpful assistant to follow the instruction"},
                    {"role": "user", "content": prompt}],
        temperature = 0,
    )

    return response["choices"][0]["message"]["content"]
    

def check_logic_accuracy(scenairo, labels, predictions): 

    results = []
    for label, prediction in zip(labels, predictions):
        try: 
            result = int(check_logic(scenairo, label=label, prediction=prediction)) 
            results.append(result)
        except:
            logger.error("Format error happens. ***********")

    return results