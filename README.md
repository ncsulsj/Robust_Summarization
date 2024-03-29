# Towards a Robust Retrieval-Based Summarization System

This repo includes the original implementation of **Towards a Robust Retrieval-Based Summarization System**.

In this implementation, we encapsulate following (1) How to generate data through **SummRAG** (data_generation directory) (2) How to train Mistral 7B instruct v0.1 with LoRA on our curated dataset (model_training directory) (3) How to evaluate the model's performance with **LogicSumm** (model_validation directory) into this repo. 


## Content 

1. [Our generated training data and trained model weights](#Data_and_Model_Weights)
2. [Model Functionality](#Model_Functionality)
3. [Inference Example](#Inference_Example)
4. [API Calling](#API_Calling)





## Data and Model Weights

Our generated training, validation data and trained model weights are available through [Data](https://huggingface.co/datasets/zycjlsj123/ragsummdata) and [Lora_Model weights](https://huggingface.co/zycjlsj123/rag_summ). 

## Model_Functionality

Our model has been finetuned to address following tasks: 

(1) Users input a topic and seek to fetch relevant text for summarization purposes.

(2) Users aim to summarize directly on their provided text without the need of retrieval.

(3) Users submit their own text and fetch supplementary text to enhance their original content before summarization.

(4) Users provide a topic and seek to obtain multiple relevant texts to do the summarization. 

In (1), our model is capable of discerning the relevance or irrelevance of retrieved texts in relation to the user's specified topic. This includes not only identifying texts that diverge in overall subject matter but also distinguishing between different subtopics. For instance, if the user's focus is on the application of ChatGPT within the financial sector, the model can differentiate texts that, while related to ChatGPT, may cover introductory aspects or applications in unrelated fields such as education, thereby ensuring alignment with the user's specific subtopic of interest. 

For (2), our model is designed to directly summarize the text provided by the user, without the need for retrieving additional external content.

## Inference_Example



In the (1) case, users could use following prompt: 

```
[INST] You are a summarization assistant to retrieve the text based on user's topic and then do the summarization. Hi, could you provide a summary of xxx ? 
[/INST] Here is the retrieval text: Start of the retrieval text: xxx End of the retrieval text.
```
In the (2) case, users could use following prompt:
```
[INST] You are a summarization assistant to do the summarization based on user's text. Hi, could you provide a summary of this text regarding xxx? 
```
In the (3) case, users could use following prompt: 
```
[INST] You are a summarization assistant to decide if combining the retrieval text with user's text to do the summarization based on its relevancy. Hi, could you summarize the following text for me? Besides, could you also check retrieve some related text and see if it can improve the summarization and also check the information conflict? [/INST]
```
In the (4) case, users could use following code to directly get the final summarization:
```
Use the function, inference_template_s7, located in model_validation/llm_utils.py, the input argument is user's topic, all the retrieval texts as a python list of string and the lora repo.
```

## API_Calling

Our model also has the ability to call the API to get the online information from many platforms such as Reddit after it finishs the summarization. [Here is the url of the API](https://www.csc2.ncsu.edu/faculty/healey/social-media-viz/production/)
