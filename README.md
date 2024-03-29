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

Our model can tell if the texts it finds match the user's topic, even spotting the difference between main topics and smaller, specific ones. For example, if you're interested in how ChatGPT is used in finance, it can identify texts that are just introductions or about using ChatGPT in education as irrelevant.

(2) Users aim to summarize directly on their provided text without the need of retrieval.

Our model, in this case, is designed to directly summarize the text provided by the user, without the need for retrieving additional external content.

(3) Users submit their own text and fetch supplementary text to enhance their original content before summarization.

Our model not only spots when subtopics don't match between the user's text and the text it retrieves, as mentioned in (1), but also 
finds the information conflict between user's text and the retrieval text and then ask user to verify it. If the retrieval text and user's text are relevant and there is no information conflict. It will combine them to derive the final summarization.  

(4) Users provide a topic and seek to obtain multiple relevant texts to do the summarization. 

Our model can summarize documents individually, using the context outlined in the paper. At each step of summarization, it blends the context with the current document to craft the summary, while also having the capacity to identify and disregard irrelevant text.Our model can summarize documents individually, using the context outlined in the paper. At each step of summarization, it blends the context with the current document to craft the summary, while also having the capacity to identify and disregard irrelevant text.

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
