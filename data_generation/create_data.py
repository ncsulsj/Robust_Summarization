# Scripts to create dialogue for each scenario 
# pypi packages 
from datasets import load_dataset
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import pickle
import random 
import openai 
from dotenv import load_dotenv
import os 
from retry import retry 
from tqdm import trange
import torch 
import ast 
from concurrent.futures import (
    as_completed,
    ProcessPoolExecutor,
)
import logging
import argparse

# local packages
from multi_docs_generate import format_conversation
from single_doc_prompt import class_dict
from utils import (gpt_create, 
                   negate_fact, 
                   create_text_scenario_7, 
                   create_text_scenario_5)
# load the topics that GPT-4 turbo can generate the factual story on (these topics are obtained from GPT-3.5 turbo by prompting ```Generate 100 topics you can generate factual stories on, return me as Python list of strings````)
from gpt_topics import topics_gpt

# initialize environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
torch.set_grad_enabled(False)
logger = logging.getLogger(__name__) 


#### =================> The following code is used to simulate a simple retrieval process of RAG to get the potenial user_text or retrieval text. 
#### =================> One could try another way of reproducing this process. 

# load the dataset, topics_collection, and the retrieval model along with its tokenizer (topics_collection is obtained from using GPT-3.5 turbo to derive the topics with at most 5 words
# for each article in the dataset we load)

ds_cnn = load_dataset("cnn_dailymail", "3.0.0")["validation"]
ds_cnn.load_faiss_index("embeddings", "cnn.faiss")
ds_cnn = ds_cnn.rename_column("article", "document")

ds_xsum = load_dataset("EdinburghNLP/xsum")["validation"]
ds_xsum.load_faiss_index("embeddings", "xsum.faiss")

q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

with open("topics", "rb") as fp:
    topic_dict = pickle.load(fp)

topic_dict_cnn = {k:v for k, v in topic_dict.items() if v == "cnn"}
topic_dict_xsum = {k:v for k, v in topic_dict.items() if v == "xsum"}
cnn_topics = list(topic_dict_cnn.keys())
xsum_topcis = list(topic_dict_xsum.keys())

def retrieval_text(question, ds, top_docs = 1):
    """
    Retrieve the text based on the question 
    """
    question_embedding = q_encoder(**q_tokenizer(question, truncation = True, return_tensors = "pt"))[0][0].numpy()
    _, retrieval_examples = ds.get_nearest_examples("embeddings", question_embedding, k = top_docs)

    return retrieval_examples["document"]

def create_dialogue_once(type, top_docs = 1):
    """
    Generate dialogue once for each scenario 
    """
    p = random.random() 
    if p <= 0.5:
        ds = ds_cnn
        topic = random.choice(cnn_topics)
        diff_topic = random.choice(cnn_topics) 
    else:
        ds = ds_xsum
        topic = random.choice(xsum_topcis)
        diff_topic = random.choice(xsum_topcis)
    sam = random.randrange(len(ds))
    user_text = ds[sam]["document"]
    another_text = ds[sam+1]["document"]
    question_topic = f"Could you find the related document regarding {topic}"
    question_diff_topic = f"Could you find the related document regarding {diff_topic}"
    question_user_text = f"could you find the related documents regarding: {user_text}"

    match type:
        case "not_relevant_online":
            retrieval_text_ = retrieval_text(question_diff_topic, ds)[0]
            prompt = class_dict[type].scenario_1_create(topic, retrieval_text_)
        case "relevant_diverse":
            ### Note this case may not return the correct doc even if you gives the topic (You need to manually check the generated dialogue.)
            retrieval_text_ = retrieval_text(question_topic, ds)[0]
            prompt = class_dict[type].scenario_2_create(topic, retrieval_text_)
        case "no_retrieve_own":
            prompt = class_dict[type].scenario_3_create(user_text)
        case "retrieve_own_not_relevant":
            prompt = class_dict[type].scenario_4_create(user_text, another_text)
        case "retrieve_own_augument":
            prompt = class_dict[type].scenario_5_create(create_text_scenario_5(random.choice(topics_gpt)))
        case "retrieve_own_conflict":
            neg_fact = negate_fact(user_text)
            prompt = class_dict[type].scenario_6_create(user_text, neg_fact)
        case "step_by_step":
            create_num = random.randint(3, top_docs - 1) # This is to create the number of factual texts based on the topic
            fact_docs = create_text_scenario_7(topic, num_create = create_num)
            docs = retrieval_text(question_diff_topic, ds, top_docs= top_docs - len(fact_docs)) # This is to create the irrelevant retrieval text
            docs += fact_docs
            random.shuffle(docs)
            return format_conversation(topic, docs)
        case _:
            prompt = ""
        
    example = gpt_create(prompt)

    return example

def create_multiple_dialogues_multiprocess(type, max_workers, num_data = 1):
    """
    Generate data for each scenario with multiple processes 
    """
    examples = []
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        future_to_result = [executor.submit(create_dialogue_once, type, top_docs = random.randint(a = 3, b = 6)) for _ in range(num_data)]
        for fut in as_completed(future_to_result):
            try:
                result = fut.result()
            except:
                logger.error("Error happens during generating dialogue")
            else:
                try:
                    examples.append(result)
                    logger.info(f"Successfully generated {len(examples)} dialogues")
                except:
                    logger.error("Error happens during appending the result")
    logger.info(f"Successfully generated {len(examples)} dialogues in total")

    return examples 


def parse_args():

    sec_choices = ["not_relevant_online", "relevant_diverse", "no_retrieve_own", "retrieve_own_not_relevant", "retrieve_own_augument", "retrieve_own_conflict", "step_by_step"]
    parser = argparse.ArgumentParser(description = "parser for generating dialogue")
    parser.add_argument("--type", type = str, help = "The type of scenario to generate dialogue", choices = sec_choices)
    parser.add_argument("--max_workers", type = int, help = "The number of workers to generate dialogue")
    parser.add_argument("--num_data", type = int, help = "The number of data to generate")
    args = parser.parse_args()

    return args 

if __name__ == "__main__":

    args = parse_args()
    diaglogues = create_multiple_dialogues_multiprocess(args.type, args.max_workers, args.num_data)
    with open(f"{args.type}_dialogues", "wb") as fp:
        pickle.dump(diaglogues, fp)