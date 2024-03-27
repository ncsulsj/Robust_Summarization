### Script to store the utility function 
# pypi packages 
import os 
import logging 
from retry import retry
import openai 
import ast 

# initalize environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

# utility functions
@retry(tries=2, delay=5)
def gpt_create(prompt, scenario):

    logger.info(f"Generating dialogue for scenario {scenario} ------------ ")
    response = openai.ChatCompletion.create(
        model = "gpt-4-1106-preview",
        messages = [{"role": "system", "content": "You are a helpful assistant to construct the conversation following the instruction"},
                    {"role": "user", "content": prompt}],
        temperature = 0.3,
    )

    return response["choices"][0]["message"]["content"]

@retry(tries=2, delay = 5)
def negate_fact(fact):

    logger.info("Negating the fact ------------ ")
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "system", "content": """You are a helpful assistant to rephrase the text in different words and diversely negate some facts or change some number in the provided text. Also, you need to add around 6 sentences extra information based on your own knowledge.
                     For example, if the date is Aug, 08 in the original text, you could change it to Sep, 10 in your genertion; if the stock price is going up, you could change it to going down; 
                     if the place where the event happens is in Beijing, you could change it to Shanghai. In summary, you need to reverse the fact in a explicit format"""},
                    {"role": "user", "content": fact}],
        temperature = 0.3,
    )

    return response["choices"][0]["message"]["content"]

def create_text_scenario_7(topic, num_create = 3):

    logger.info(f"Creating {num_create} factual texts based on the topic {topic} for scenario 7 ------------ ")
    response = openai.ChatCompletion.create(
        model = "gpt-4-1106-preview",
        messages = [{"role": "system", "content": f"You are a helpful assistant to create {num_create} factual diverse texts based on the provided topic. These texts,although,discuss the same topic, they must discuss completely different events regarding this topic"},
                    {"role": "user", "content": f"""Here is the provided topic: {topic} \n 
                     You MUST return me as a list of texts like ["text1", "text2", ...]. Each text must be around 300 english words. TRY TO WRTIE AS LONG AS POSSIBLE"""}],
        temperature = 0.5,
    )
    rs = ast.literal_eval(response["choices"][0]["message"]["content"])

    return rs

def create_text_scenario_5(topic):

    logger.info(f"Creating factual texts based on the topic {topic} for scenario 5 ------------ ")
    prompt = """
    You are asked to generate ()factual user text and factual retrieval text regarding topic {0} which discuss the same subtopic (the same event) and they help each other for the text summarization 
    for either of these two texts.\n The subtopic you generate for each of them must be specific instead of the introduction to the topic.
    Return me the text with following template:\n
    ##
    Here is the user's text:\n
    <user_text>\n
    Here is the retrieval text:\n
    <retrieval_text>\n
    ##
    """.format(topic)

    response = openai.ChatCompletion.create(
        model = "gpt-4-1106-preview",
        messages = [{"role": "system", "content": "You are a helpful assistant to construct user text and the retrieval text following the instruction"},
                    {"role": "user", "content": prompt}],
        temperature = 0.5,
    )

    return response["choices"][0]["message"]["content"]