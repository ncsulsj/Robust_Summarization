# Towards a Robust Retrieval-Based Summarization System

This includes the original implementation of **Towards a Robust Retrieval-Based Summarization System** 

In this implementation, we encapsulate following (1) How to generate data through **SummRAG** (data_generation directory) (2) How to train Mistral 7B instruct v0.1 with LoRA on our curated dataset (model_training directory) (3) How to evaluate the model's performance with **LogicSumm** (model_evaluation directory) into this repo. 


## Content 
1. [Our generated training data and trained model weights](#data_model_weights)
2. [Inference Example](#Inference_Example)





## data_model_weights
Our generated training, validation data and trained model weights are available through [Data](https://huggingface.co/datasets/zycjlsj123/ragsummdata) and [Model weights](https://huggingface.co/zycjlsj123/rag_summ). 


## Inference_Example
Our model is finetuned to deal with following tasks: 

(1) Users provide a topic and aims to retrieve a text to do summarization.

(2) Users want to summarize only on their own text (no need for retrieval)

(3) Users provide their own text and aims to retrieve a text to augument their own text to do summarization (the retreival text and user's text may exist information conflict, irrelevancy).

(4) Users want to retrieve multiple documents and summarize them robustly, i.e. robustly to deal with the irrelevant texts in the retrieval texts. 

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
