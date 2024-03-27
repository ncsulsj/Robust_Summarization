### This script is used to evaluate the performance of different summarization methods on the multi-docs test data.
### pypi packages 
from langchain.chains.summarize import load_summarize_chain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from datasets import load_dataset
import pickle 
import random 
import re
from evaluate import load
from langchain.docstore.document import Document
from huggingface_hub import snapshot_download
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
import argparse
import logging

### local packages
from inference_utils import *
from llm_utils import *

### initialize environment
mistral_api_key = "1234"
chat = ChatMistralAI(mistral_api_key=mistral_api_key, model = "open-mistral-7b")
bertscore = load("bertscore")
rouge = load("rouge")
logger = logging.getLogger(__name__)

# utility functions
def final_summarize_data(true_data, mocure_data, total_docs):

    all_texts = list(set(pick_all_retrieval_texts(true_data))) 
    topic = pick_user_topic(true_data[0])
    pattern = r"The final summarization is: (.*)"
    match = re.search(pattern, true_data[-1])
    if match:
        res = match.group(1)
        print('kk')
    else:
        res = None
    ra_numbers = total_docs - len(all_texts)
    ra_texts = mocure_data.select(random.choices(range(3800, 4800), k = ra_numbers))["article"]
    all_texts += ra_texts
    random.shuffle(all_texts)
    
    return {"topic":topic, "texts": all_texts, "label": res}

def split_list_including_stop_sign_variants(input_list):
    result = []
    current_sublist = []

    for item in input_list:
        if "You are a summarization assistant" in item:
            if current_sublist:
                result.append(current_sublist)
            current_sublist = [item]
        else:
            current_sublist.append(item)
    if current_sublist:
        result.append(current_sublist)

    return result


################### testing 

def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate the performance of different summarization methods on multi-docs test data")
    parser.add_argument("--case", type = str, default = "stuff", help = "the summarization method to evaluate")
    parser.add_argument("--data_path", type = str, default = "stuff", help = "the data to be evaluated")
    parser.add_argument("--lora_repo", type = str, help = "the lora repo of the trained model")
    args = parser.parse_args()

    return args
    
    
def main(case = "stuff"):

    args = parse_args()

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    test = dataset["test"]
    with open(args.data_path, "rb") as fp:
        egs = pickle.load(fp)
    sub_egs = split_list_including_stop_sign_variants(egs)
    correct_egs = []
    for eg in sub_egs:
        try:
            all_texts = pick_all_retrieval_texts(eg)
            correct_egs.append(eg)
        except:
            logger.info("removing data without correct format")

    final_test_data = [ final_summarize_data(sub_eg, test, total_docs =10) for sub_eg in correct_egs]
    correct_test_data = [] 

    for data in final_test_data:
        data["texts"] = [Document(page_content = doc) for doc in data["texts"]]
        correct_test_data.append(data)

    # 1. stuff summarization
    if case == "stuff":
        document_prompt = PromptTemplate(
            input_variables = ["page_content"],
            template = "{page_content}",
        )
        chain_prompt = PromptTemplate(
            input_variables = ["context", "topic"],
            template = """
            Write a summary of the following text regarding topic {topic} and skip irrelevant text with respect to this topic. \n
            Here is the text: {context}
            """
        )
        llm_chain = LLMChain(llm = chat, prompt = chain_prompt)
        stuff_chain = StuffDocumentsChain(llm_chain = llm_chain, document_prompt = document_prompt,  document_variable_name= "context" )

        results = [stuff_chain.invoke({"topic": data["topic"], "input_documents": data["texts"]})["output_text"] for data in correct_test_data] 
        labels = [data["label"] for data in correct_test_data]

        with open("last_1.pkl", "wb") as fp:
            pickle.dump(results, fp)
        with open("last_2.pkl", "wb") as fp:
            pickle.dump(labels, fp)


    # 2. map reduce summarization 
    if case == "map":
        map_prompt_template = """
                            Write a summary of this chunk of text regarding topic {topic} that includes the main points and any important details (skip irrelevant text with respect to this topic.).
                            {text}
                            """
        map_prompt = PromptTemplate(template = map_prompt_template, input_variables = ["topic", "text"])

        combine_prompt_template = """
                Write a concise summary of the following text delimited by triplet backquotes. ```{text}```
                Here is your summary: 
        """
        combine_prompt = PromptTemplate(template = combine_prompt_template, input_variables = ["text"])
        map_reduce_chain = load_summarize_chain(
            chat,
            chain_type = "map_reduce",
            map_prompt = map_prompt, 
            combine_prompt = combine_prompt,
            return_intermediate_steps = False
        )

        results = [map_reduce_chain.invoke({"topic": data["topic"], "input_documents": data["texts"]})["output_text"] for data in correct_test_data] 
        labels = [data["label"] for data in correct_test_data]

        with open("last_1.pkl", "wb") as fp:
            pickle.dump(results, fp)
        with open("last_2.pkl", "wb") as fp:
            pickle.dump(labels, fp)


    # 3. refine summarization 
    if case == "refine":

        question_prompt_template = """
            Provide a summary of the following text with respect to topic {topic} (skip irrelevant text with respect to the topic):
            TEXT: {text}
            SUMMARY: 
        """
        question_prompt = PromptTemplate(
            template = question_prompt_template, input_variables = ["topic", "text"]
        )

        refine_prompt_template = """
                    Write a concise summary of the following text delimited by triple backquotes.
                    ```{text}```
                    SUMMARY:
                    """
        refine_prompt = PromptTemplate(
            template=refine_prompt_template, input_variables=["text"]
        )

        refine_chain = load_summarize_chain(
            chat,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=False,
        )


        results = [refine_chain.invoke({"topic": data["topic"], "input_documents": data["texts"]})["output_text"] for data in correct_test_data] 
        labels = [data["label"] for data in correct_test_data]

        with open("last_1.pkl", "wb") as fp:
            pickle.dump(results, fp)
        with open("last_2.pkl", "wb") as fp:
            pickle.dump(labels, fp)


    # 4. our method 
    if case == "us":

        engine = initialize_engine()

        lora_repo = args.lora_repo

        results = [inference_template_s7(data["topic"], [i.page_content for i in data["texts"]], lora_repo = lora_repo, engine = engine) for data in correct_test_data]
        labels = [data["label"] for data in correct_test_data]

        with open("last_1.pkl", "wb") as fp:
            pickle.dump(results, fp)
        with open("last_2.pkl", "wb") as fp:
            pickle.dump(labels, fp)


if __name__ == "__main__":

    main(case = "refine")















