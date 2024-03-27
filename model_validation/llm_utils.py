### This file along with inference_utils.py contains the utility functions for the model validation. 
### pypi packages
import openai 
from dotenv import load_dotenv
import os 
import re 
import logging

### local packages
from inference_utils import * 

### initalize environment
logger = logging.getLogger(__name__)

# utility functions

def pick_user_topic(input_text):

    prompt = f"""
    You are given a text which is the conversation between user and the summarization asssitant. You need to extract the topic 
    of the user's query and copy it to me. This topic appears inside the first [INST] and [/INST].
    Here is the input text: ###{input_text}###
    """

    response = openai.ChatCompletion.create(
        model = "gpt-4-1106-preview",
        messages = [{"role": "system", "content": "You are a helpful assistant to follow the instruction"},
                    {"role": "user", "content": prompt}],
        temperature = 0,
    )

    return response["choices"][0]["message"]["content"]


def pick_retrieval_text(input_text): 

    pattern = r"Start of the retrieval text:(.*?)End of the retrieval text"
    match = re.search(pattern, input_text)
    if match:
        extracted_text = match.group(1).strip()
    else:
        logger.warn("No match found")

    return extracted_text



def pick_user_text(input_text):

    pattern = r"\[INST\](.*?)\[/INST\]"
    matches = re.findall(pattern, input_text)
    if len(matches) >= 2:
        second_instance_content = matches[1]
    else:
        logger.warn("Less than two [INST]...[/INST] blocks found")

    return second_instance_content

def pick_all_retrieval_texts(input_text_list):

    all = []
    for text in input_text_list:
        pattern = r"Start of the retrieval text:(.*?)End of the retrieval text"
        match = re.findall(pattern, text)
        if len(match) >= 2: 
            all.append(match[-1])
        else:
            all.append(match[0])

    return all 


def inference_template_s7(topic, retrieval_texts, lora_repo, engine):

    current = 0
    start = f"[INST] You are a summarization assistant to summarize the documents one by one. Could you summarize the text related to {topic}? There are total {len(retrieval_texts)} documents to summarize.[/INST]Here is the retrieval text:  Start of the retrieval text: {retrieval_texts[current]} End of the retrieval text."
    result = ""
    while "final summarization" not in result:
        prompt = create_test_prompt(lora_path =snapshot_download(repo_id= lora_repo), 
                             text_list = [start])
        result = process_requests(engine, prompt, use_lora= True)[0]["pred"]
        current += 1 
        if current < len(retrieval_texts):
            start = f" [INST] In the process of summarizing documents one by one. Here is the retrieval text: Start of the retrieval text: {retrieval_texts[current - 1]} End of the retrieval text. " + result + "[/INST]" + f"Here is the retrieval text: Start of the retrieval text: {retrieval_texts[current]} End of the retrieval text."
        elif current == len(retrieval_texts):
            start = f" [INST] In the process of summarizing documents one by one. Here is the retrieval text: Start of the retrieval text: {retrieval_texts[current - 1]} End of the retrieval text. " + result + "[/INST]"
        else:
            logger.info('loop finished ========>')
    parts = result.split("The final summarization is: ", 1)[1]

    return parts


# This is to create the prompt for the baseline LLMs model inference
def prompt_llm_scenairo_1_2(user_topic, text):

    prompt = f"""[INST]
    You are given following task based on the given text. Here is the text: ###{text}###. You first check if the text is relevant with the 
    {user_topic}. Using following answer format:\n 
    (1) if it is relveant, return me using following format "The retrieval text is relevant with user's topic. Start summarizing on it: content of the summarization" \n
    (2) if it is not relevant, return me using following format "The retrieval text is not relevant with user's topic" \n
    Here is your reply: [/INST]
    """

    return prompt 


def prompt_llm_scenairo_1_2_COT(user_topic, text):

    prompt = f"""[INST]
    You are given following task based on the given text. Here is the text: ###{text}###. You first check if the text is relevant with the 
    {user_topic}.  You are required to finish the task step by step:\n 
    The first step is to determine the relevancy of the retrieval text to the user topic. \n
    Then the second step is based on the result of the relevancy: \n 
    (1) if it is relveant, return me using following format "The retrieval text is relevant with user's topic. Start summarizing on it: content of the summarization" \n
    (2) if it is not relevant, return me using following format "The retrieval text is not relevant with user's topic" \n
    Here is your reply: [/INST]
    """

    return prompt 

def prompt_llm_scenairo_1_2_COT_instruction(user_topic, text):

    prompt = f"""[INST]
    You are given following task based on the given text. Here is the text: ###{text}###. You first check if the text is relevant with the 
    {user_topic}.  You are required to finish the task step by step:\n 
    The first step is to determine the relevancy of the retrieval text to the user topic. \n
    Then the second step is based on the result of the relevancy: \n 
    (1) if it is relveant, return me using following format "The retrieval text is relevant with user's topic. Start summarizing on it: content of the summarization" \n
    (2) if it is not relevant, return me using following format "The retrieval text is not relevant with user's topic" \n

    Here is an example: \n
    The user topic would like to know the summary about ChatGPT application in Finance. The retrieval text is ### ChatGPT is a chatbot developed by OpenAI and launched on November 30, 2022. Based on a large language model, it enables users to refine and steer a conversation towards a desired length, format, style, level of detail, and language.###. Then, you need to output it is not relevant
    since the user asks the specific finance application of ChatGPT but the retrieval text reflects the ChatGPT introduction. 
    Here is your reply: [/INST]
    """

    return prompt 


def prompt_llm_scenairo_3(user_text):

    prompt = f"""[INST]
    You are a summarization assistant to summarize following text and return ONLY the summary to me.
    Here is the text ###{user_text}### [/INST]
    """
    
    return prompt 


def prompt_llm_scenairo_4_5_6(user_text, retrieval_text):

    prompt = f"""[INST]
    You are given two tasks based on the given two texts. Here is the user text: ###{user_text}###.\n Here is the retrieval text: 
    ###{retrieval_text}###.\n You first check if the retrieval text text is relevant with the user text, and if it is relevant, 
    check if there is any information conflict between the retrieval text and the user text. Using following format:
    (1) if they are not relevant, you should return to me: the user text is not relevant with the retrieval text. Start summarizing only on user text: content of the summarization\n
    (2) if they are relevant but the retrieval text has information conflict with the user text, you only need to return "There is information conflict between the user text and the retrieval text" \n
    (3) if they are relevant and there is no information conflict between them, you should return to me: the user text is relevant with the retrieval text and there is no information conflict. Start summarizing on 
    both retrieval text and user text: content of the summarization. \n [/INST]
    """

    return prompt 


def prompt_llm_secnairo_4_5_6_COT(user_text, retrieval_text):

    prompt = f"""[INST]
    You are instructed to finish following text step by step. Here is the user text: ###{user_text}###.\n Here is the retrieval text: 
    ###{retrieval_text}###.\n 
    The first step is to check if the retrieval text is relevant with the user text. Based on the check result, you are ready to implement the following step. 
    (1) if they are not relevant, you should return to me: the user text is not relevant with the retrieval text. Start summarizing only on user text: content of the summarization\n
    (2) if they are relevant but the retrieval text has information conflict with the user text, you only need to return "There is information conflict between the user text and the retrieval text" \n
    (3) if they are relevant and there is no information conflict between them, you should return to me: the user text is relevant with the retrieval text and there is no information conflict. Start summarizing on 
    both retrieval text and user text: content of the summarization. \n [/INST]
    """

    return prompt 


def prompt_llm_secnairo_4_5_6_COT_instruction(user_text, retrieval_text):

    prompt = f"""[INST]
    You are instructed to finish following text step by step. Here is the user text: ###{user_text}###.\n Here is the retrieval text: 
    ###{retrieval_text}###.\n 
    The first step is to check if the retrieval text is relevant with the user text. Based on the check result, you are ready to implement the following step. 
    (1) if they are not relevant, you should return to me: the user text is not relevant with the retrieval text. Start summarizing only on user text: content of the summarization\n
    (2) if they are relevant but the retrieval text has information conflict with the user text, you only need to return "There is information conflict between the user text and the retrieval text" \n
    (3) if they are relevant and there is no information conflict between them, you should return to me: the user text is relevant with the retrieval text and there is no information conflict. Start summarizing on 
    both retrieval text and user text: content of the summarization. \n 

    Here is one example: The user text is ### The Ragdoll is a breed of cat with a distinct colorpoint coat and blue eyes. Its morphology is large and weighty, and it has a semi-long and silky soft coat. American breeder Ann Baker developed Ragdolls in the 1960s. They are best known for their docile, placid temperament and affectionate nature.
    ###.\n The retrieval text is ###A domestic short-haired cat is a cat possessing a coat of short fur, not belonging to any particular recognised cat breed. In the United Kingdom, they are colloquially called moggies.### Then, in this example,
    your reply should be: The user text is not relevant with the retrieval text. Start summarizing only on user text: Ragdolls are large, gentle cats with colorpoint coats and blue eyes. \n 

    [/INST]
    """

    return prompt 