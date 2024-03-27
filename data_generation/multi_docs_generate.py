### Script to create dialogues with multiple documents - Aspect 4 Scenario 7 through step-by-step generation
# pypi packages 
from dotenv import load_dotenv
import openai 
import os 
import ast 
from retry import retry 
import logging 


# local packages
from multi_docs_one_shot_example import (
    start_conversation_not_relevant,
    start_conversation_relevant,
    mid_conversation,
    end_conversation,
)

# initalize environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

# utility functions 
def finish_start_prompt(template_relevent, template_not_relevant, num_docs, first_document, topic):

    prompt = """
    You are instructed to construct the start of the conversation between the assistant and the user requires the assistant to do summarization document by document on some topic.
    Some special tokens need to be added to the conversation. You are required to follow the format of the provided example, 
    including the position of special tokens. Here are special tokens: \n
    [1] The information inside <Context> and </Context> should be No context till now (You could diversify this sentence)\n
    [2] [Relevant] and [Irrelevant] are used to check whether the retrieval text inside <paragraph> and </paragraph> are relevant with the user query.\n
    [3] Content inside <Count> and </Count> is to check how many documents left to summarize. \n
    [4] [Topic] are used to keep the topic of the user query. \n
    The summarization should be appended after </Context>. The utitliy score should be 1 if [Irrelevant]. The retrieval text at each step should be inside of <paragraph> and </paragraph>\n
    Here is a relevant example:\n
    {0}\n
    Here is a not relevant example:\n
    {1}\n
    ### Now, you are instructed to follow the above example to create the start of the convseration. There are total {2} documents, the topic is {4}, and the first document is following:\n
    {3}\n
    ### The response must only be a list of four dictionaries without saying any other things.
    """.format(template_relevent, template_not_relevant, num_docs, first_document, topic)

    return prompt 

def finish_mid_prompt(start_piece, mid_conversation_relevant, doc):

    prompt =  """
    You are instructed to construct the conversation between the assistant itself and its goal is to do summarization document by document on some topic. Some sepcial tokens need to be added to the convseration.
    You are required to follow the format of the provided example, including the position of special tokens. Here are special tokens: \n
    [1] The information inside <Context> and </Context> is the context you need to rely on when you do the summarization by combining with the retrieval text.\n
    [2] [Relevant] and [Irrelevant] are used to check whether the retrieval text inside <paragraph> and </paragraph> are relevant with the user query.\n
    [3] Content inside <Count> and </Count> are to check how many documents left to summarize. \n
    [4] [Topic] are used to keep the topic of the user query. \n
    Here is one example:\n
    {1}\n
    ##Now, I will provide you with the first piece of the conversation. You need to keep it UNCHANGED. Here is the first piece of the convseration: \n
    {0}##\n
    and here is the new retrieval text:\n
    {2}##\n
    ##Construct the new piece of the conversation: Context should keep unchanged if [Irrelevant] appears on the first piece of conversation and need to be changed to the summarization in the first piece if the [Relevant] appears in the first piece of conversation.
    If the new retrieval text is still irrelevant to the user query, the summarization should be same as the context; if it is relevant, then the summarization should consider both the content of context and the retrieval text (DO NOT LOSE ANY INFORMATION IN THE CONTEXT)##\n
    ##The position of summarization should be appended after </Context>\n!!!!(DO NOT LOSE ANY INFORMATION IN THE CONTEXT EVEN EXTENDING THE LENGTH OF THE SUMMARIZATION. IT IS VERY IMPORTANT)!!!! \nYou MUST RETURN ME A LIST OF TWO DICTIONARIES WITHOUT SAYING ANY OTHER THINGS##
    """.format(start_piece, mid_conversation_relevant, doc)

    return prompt 

def finish_last_prompt(start_piece, final_conversation_relevant):

    prompt = """
    You are instructed to construct the final step of the onversation between the assistant itself and its goal is to do summarization document by document on some topic. Some sepcial tokens need to be added to the convseration.
    You are required to follow the format of the provided example, including the position of special tokens. Here are special tokens: \n
    [1] The information after [Context] should be the information you should not forget when you do the summarization.\n
    [2] [Relevant] and [Irrelevant] are used to check whether the retrieval text inside <paragraph> and </paragraph> are relevant with the user query.\n
    [3] Content inside <Count> and </Count> are to check how many documents left to summarize. \n
    [4] [Topic] are used to keep the topic of the user query. \n
    Here is one example:\n
    {0}\n
    ##Now, I provide you with the first piece of convseration. You need to keep it UNCAHNGED. Here is the first piece of the conversation: \n
    {1}\n##
    ##All you need to do is to generate next piece of conversation. If [Irrelvant] appears in the first piece, the final summarization is the context; if [Relevant] appears, the final summarization is the summarization from last part, which is the content after </Context> and before [Utility]##
    You need to return me both the first piece and your generated conversation. You MUST RETURN ME A LIST OF TWO DICTIONARIES WITHOUT SAYING ANY OTHER THINGS.
    """.format(final_conversation_relevant, start_piece)

    return prompt 

@retry(tries=2, delay=5)
def gpt_create(prompt):

    logger.info(f"start generating conversation for Aspect 4 scenario 7")
    response = openai.ChatCompletion.create(
        model = "gpt-4-1106-preview",
        messages = [{"role": "system", "content": "You are a helpful assistant to construct the conversation following the instruction"},
                    {"role": "user", "content": prompt}],
        temperature = 0.3,
    )

    return response["choices"][0]["message"]["content"]

def format_conversation(topic, docs):

    num_docs = len(docs)
    first_document = docs[0]
    first_conversation_prompt = finish_start_prompt(start_conversation_relevant, start_conversation_not_relevant, num_docs, first_document, topic)
    conv = ast.literal_eval(gpt_create(first_conversation_prompt))
    start_piece = conv.pop(-1)
    for i in range(1, len(docs)):
        mid_prompt = finish_mid_prompt(start_piece, mid_conversation, docs[i])
        res = ast.literal_eval(gpt_create(mid_prompt))
        start_piece = res.pop(-1)
        conv += res 
    last_prompt = finish_last_prompt(start_piece, end_conversation)
    conv += ast.literal_eval(gpt_create(last_prompt))

    return conv