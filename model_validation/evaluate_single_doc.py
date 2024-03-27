# This script contains the code to evaluate the performance of a LLM on the scenario from 1 to 6
### pypi packages
import pickle
from typing import Optional, List, Tuple 
from huggingface_hub import snapshot_download
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
import time 
import argparse
import logging
from abc import ABC 
import openai 
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import ast 
import boto3 
import os 
import json 

### local packages
from llm_utils import * 
from inference_utils import * 


### initializing environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
brt = boto3.client(service_name = "bedrock-runtime", region_name = "us-east-1")
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class LLM(ABC):

    def __init__(self):

        pass 
    
    def predict(self, prompt):

        raise NotImplementedError 
    

class GPT(LLM):

    def __init__(self, model_name):

        self.model_name = model_name
    
    def predict(self, prompt):

        response = openai.ChatCompletion.create(
            model = self.model_name, 
            messages = [{"role": "system", "content": "You are a helpful summarization assistant"}, 
                        {"role": "user", "content": prompt}],
                        temperature = 0.1,
        )
        answer = response["choices"][0]["message"]["content"]
    
        return answer
    
    def __str__(self):

        return "GPT"

class Claude2(LLM):

    def __init__(self):

        pass
    
    def predict(self, prompt):

        body = json.dumps({
        "prompt": "\n\nHuman: {}\n\nAssistant:".format(prompt),
        "max_tokens_to_sample": 800,
        "temperature": 0.1,
        "top_p": 0.9,
        })
        modelId = 'anthropic.claude-v2'
        accept = 'application/json'
        contentType = 'application/json'
        response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        return response_body.get('completion')
    
    def __str__(self):

        return "Claude2"
    
class Mixtral(LLM):

    def __init__(self):

        self.engine = initialize_engine()

    def predict(self, prompt):

        prompt = create_test_prompt(lora_path =snapshot_download(repo_id="zycjlsj123/try_fifth"), 
                             text_list = [prompt])
        result = process_requests(self.engine, prompt, use_lora= False)

        return result[0]["pred"]
    def __str__(self):

        return "Mixtral"
        


class Jurassic(LLM):

    def __init__(self):
        pass

    def predict(self, prompt):
        
        body = json.dumps({
        "prompt": "\n\nHuman: {}\n\nAssistant:".format(prompt),
        "maxTokens": 800,
        "temperature": 0.1,
        "topP": 0.9
        })
        modelId = 'ai21.j2-ultra-v1'
        accept = 'application/json'
        contentType = 'application/json'
        response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        
        return response_body.get('completions')[0].get('data').get('text')
    
    def __str__(self):

        return 'Jurassic'
    
class Llama2(LLM):

    def __init__(self):

        pass 

    def predict(self, prompt):

        body = json.dumps({
        "prompt":  "\n\nHuman: {}\n\nAssistant:".format(prompt),
        "max_gen_len": 800,
        "temperature": 0.1,
        "top_p": 0.9,
        })
        modelId = 'meta.llama2-13b-chat-v1'
        accept = 'application/json'
        contentType = 'application/json'
        try:
            response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

            return json.loads(response.get('body').read())['generation']
        except:

            return ""

    def __str__(self):

        return "Llama"

    
def llm_predicts(Llm, scenairo_type, texts, labels, prompt_type):

    if scenairo_type == 1:
        topics = [pick_user_topic(data) for data in texts]
        re_texts = [pick_retrieval_text(data) for data in texts]
        if prompt_type == "direct":
            prompts = [prompt_llm_scenairo_1_2(topic, text) for topic, text in zip(topics, re_texts)]
        elif prompt_type == "zero_shot":
            prompts = [prompt_llm_scenairo_1_2_COT(topic, text) for topic, text in zip(topics, re_texts)]
        elif prompt_type == "one_shot":
            prompts = [prompt_llm_scenairo_1_2_COT_instruction(topic, text) for topic, text in zip(topics, re_texts)]
        else:
            raise ValueError("No such prompt type exists ========>")
        results = [Llm.predict(prompt) for prompt in prompts]
        with open("results.pkl", "wb") as fp:
            pickle.dump(results, fp)
        perfs = check_logic_accuracy(scenairo = scenairo_type, labels = labels, predictions = results)
        logging.info(f"The logic accuracy for {Llm} under scenairo {scenairo_type} {np.mean(perfs)}")
    elif scenairo_type == 2: 
        topics = [pick_user_topic(data) for data in texts]
        re_texts = [pick_retrieval_text(data) for data in texts]
        if prompt_type == "direct":
            prompts = [prompt_llm_scenairo_1_2(topic, text) for topic, text in zip(topics, re_texts)]
        elif prompt_type == "zero_shot":
            prompts = [prompt_llm_scenairo_1_2_COT(topic, text) for topic, text in zip(topics, re_texts)]
        elif prompt_type == "one_shot":
            prompts = [prompt_llm_scenairo_1_2_COT_instruction(topic, text) for topic, text in zip(topics, re_texts)]
        else:
            raise ValueError("No such prompt type exists ========>")
        results = [Llm.predict(prompt) for prompt in prompts]
        perfs = check_logic_accuracy(scenairo = scenairo_type, labels = labels, predictions = results)
        logging.info(f"The logic accuracy for {Llm} under scenairo {scenairo_type} is {np.mean(perfs)}")
    elif scenairo_type == 3: 
        user_texts = [pick_user_text(data) for data in texts]
        prompts = [prompt_llm_scenairo_3(data) for data in user_texts]
        results = [Llm.predict(prompt) for prompt in prompts]
        coherence_bert = check_coherence_quality(metric = "bertscore", labels= labels, predictions = results)
        coherence_rouge = check_coherence_quality(metric = "rouge", labels= labels, predictions = results)
        logging.info(f"The bert score for {Llm} under scenairo {scenairo_type} is {coherence_bert}")
        logging.info(f"The rouge score for {Llm} under scenairo {scenairo_type} is {coherence_rouge}")
    elif scenairo_type == 4 or scenairo_type == 5 or scenairo_type == 6:
        user_texts = [pick_user_text(data) for data in texts]
        re_texts = [pick_retrieval_text(data) for data in texts]
        if scenairo_type == 4:
            with open("user_texts.pkl", "rb") as fp:
                user_texts += pickle.load(fp)
            with open("ree_texts.pkl", "rb") as fp:
                re_texts += pickle.load(fp)
        if prompt_type == "direct":
            prompts = [prompt_llm_scenairo_4_5_6(user_text, re_text) for user_text, re_text in zip(user_texts, re_texts)]
        elif prompt_type == "zero_shot":
            prompts = [prompt_llm_secnairo_4_5_6_COT(user_text, re_text) for user_text, re_text in zip(user_texts, re_texts)]
        elif prompt_type == "one_shot":
            prompts = [prompt_llm_secnairo_4_5_6_COT_instruction(user_text, re_text) for user_text, re_text in zip(user_texts, re_texts)]
        else:
            raise ValueError("No such prompt type exists ========>")
        results = [Llm.predict(prompt) for prompt in prompts]
        perfs = check_logic_accuracy(scenairo = scenairo_type, labels = labels, predictions = results)
        logging.info(f"The logic accuracy for {Llm} under scenairo {scenairo_type} is {np.mean(perfs)}")
        coherence_bert = check_coherence_quality(metric = "bertscore", labels= labels, predictions = results)
        coherence_rouge = check_coherence_quality(metric = "rouge", labels= labels, predictions = results)
        logging.info(f"The bert score for {Llm} under scenairo {scenairo_type} is {coherence_bert}")
        logging.info(f"The rouge score for {Llm} under scenairo {scenairo_type} is {coherence_rouge}")
    else:
        raise ValueError(f"This scenairo {scenairo_type} has not been implemented ======>")


def split_list(lst, size):

    for i in range(0, len(lst), size):
        
        yield lst[i: i + size]

def parse_args(): 

    parser = argparse.ArgumentParser(description = "Evaluating LLMs on the constructed summarization task")

    parser.add_argument("--data", type = str, help = "The path of the dataset to be used")
    parser.add_argument("--lora_path", type = str, help = "The huggingface lora repo path")
    parser.add_argument("--scenairo", type = int, help = "the scenairo we want to test")
    parser.add_argument("--batch_size", type = int, help = "the parallel inference batch size")
    parser.add_argument("--llm_type", type = str, help = "The specific LLM we want to testsa")
    parser.add_argument("--use_lora", type = bool, help = "use lora or not")
    parser.add_argument("--prompt_type", type = str, help = "The type of the prompts to use")
    
    args = parser.parse_args() 

    return args 


def main():

    args = parse_args() 

    start_time = time.time()
    logging.info("Evaluation Begins ============================ >>>>>>")

    try:
        with open(args.data, "rb") as fp: 
            test_data = pickle.load(fp)
    except Exception as e: 
        logging.info("A data import error occured")
    
    inps = []; labels = [] 

    for text in test_data:
        inp, label = split_input_and_label(text, scenairo_type = args.scenairo)
        if inp != "" and label != "":
            inps.append(inp)
            labels.append(label)

    if args.llm_type == "rag_llm":
        
        inp_chunks = list(split_list(inps, size = args.batch_size))
        label_chunks = list(split_list(labels, size = args.batch_size))

    
        final_results = []; preds = []
        
        logging.info(f"There are total {len(inps)} examples to be tested")

        engine = initialize_engine()

        for inp_batch, label_batch in zip(inp_chunks, label_chunks):
            prompts = create_test_prompt(lora_path =snapshot_download(repo_id= args.lora_path), 
                                text_list = inp_batch, 
                                )
            results = process_requests(engine, prompts, args.use_lora)
            results = sorted(results, key = lambda x: int(x["idx"]))
            results = [result["pred"] for result in results]
            preds.append(results)
            final_results.append(check_logic_accuracy(scenairo = args.scenairo, labels = label_batch, predictions = results)) 
        final_results = [num for chunk in final_results for num in chunk]
        preds = [pred for chunk in preds for pred in chunk]

        with open("result.pkl", "wb") as fp:
            pickle.dump(preds, fp)

        logging.info(f"There are total {len(final_results)} examples which have been tested")

        accuracy = np.mean(final_results) 

        coherence_bert = check_coherence_quality(metric = "bertscore", labels= labels, predictions = preds)
        coherence_rouge = check_coherence_quality(metric = "rouge", labels= labels, predictions = preds)
        
        
        logging.info(f"The logic accuracy for {args.llm_type} under scenairo {args.scenairo} is {accuracy}")
        logging.info(f"The bert score is for {args.llm_type} under scenairo {args.scenairo} is {coherence_bert}")
        logging.info(f"The rouge score is for {args.llm_type} under scenairo {args.scenairo} is {coherence_rouge}")
    
    else:

        match args.llm_type:
            case "Claude2":
                Llm = Claude2() 
            case "Jurassic":
                Llm = Jurassic()
            case "Llama":
                Llm = Llama2()
            case "GPT":
                Llm = GPT("gpt-3.5-turbo") 
            case "Mixtral": 
                Llm = Mixtral()
            case _:
                raise ValueError("The LLM is not supported")
        
        llm_predicts(Llm, args.scenairo, test_data, labels, args.prompt_type)

if __name__ == "__main__":
    main()