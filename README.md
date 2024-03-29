# Towards a Robust Retrieval-Based Summarization System

Welcome to the official repository for **Towards a Robust Retrieval-Based Summarization System**. This repository hosts the original implementation of our robust system designed to enhance retrieval-based summarization processes.

## Overview

The system is structured into distinct components to streamline the process of generating data, training the model, and evaluating performance. Specifically, it includes:

- **Data Generation**: Utilizing **SummRAG** to create datasets. (See `data_generation` directory)
- **Model Training**: Instructions on training the Mistral 7B instruct v0.1 model with LoRA on our curated datasets. (Located in `model_training` directory)
- **Model Evaluation**: How to assess the model's performance using **LogicSumm**. (Found in `model_validation` directory)

## Contents

1. [Data and Model Weights](#data-and-model-weights)
2. [Model Functionality](#model-functionality)
3. [Example Inference](#example-inference)
4. [API Integration](#api-integration)

## Data and Model Weights

Access our meticulously generated training and validation datasets, along with the LoRA model weights, at the following locations:
- Training and Validation Data: [Hugging Face Datasets](https://huggingface.co/datasets/zycjlsj123/ragsummdata)
- Model Weights: [LoRA Model Weights](https://huggingface.co/zycjlsj123/rag_summ)

## Model Functionality

Our model is fine-tuned for various summarization tasks:
1. **Topic-based Text Retrieval and Summarization**: Capable of identifying the relevancy of retrieval text to a user-defined topic (not only the topic but also the subtopic. i.e. ChatGPT application in Finance is not relevant with ChatGPT introduction or application in Education)
2. **Direct Text Summarization**: Offers summarization on user-provided texts without external text retrieval.
3. **Enhanced Summarization with Supplementary Text**: Identifies and integrates relevant supplementary texts with the original content for comprehensive summarization (also takes care of the case of different subtopic). It also can identify information conflict between them.
4. **Multi-document Summarization**: Efficiently summarizes multiple documents, filtering out irrelevant content for coherent summaries.

## Example Inference

To utilize our model for different use cases, consider the following prompts:

- **For Topic-based Summarization**:
