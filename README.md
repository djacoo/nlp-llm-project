# Email Routing with Large Language Models

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-Report-008080?logo=latex&logoColor=white)

NLP 2025-26 project for the LLM module at the University of Verona. We build and compare three LLM-based agents that automatically route customer support emails to one of five departments (Technical Support, Customer Service, Billing and Payments, Sales and Pre-Sales, General Inquiry). The agents are evaluated on the [customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) dataset filtered to English emails, split into train/validation/test sets.

| Agent | Approach | Model | Accuracy |
|-------|----------|-------|----------|
| 1 - Frozen Prompting | Zero-shot log-likelihood scoring, no weight updates | GPT-2, DistilGPT-2 | 22.7%, 28.4% |
| 2 - LoRA Fine-Tuning | Parameter-efficient adaptation (~1% trainable params) | GPT-2, DistilGPT-2 | 70.7%, 70.4% |
| 3 - DistilBERT Classifier | Supervised sequence classification, full fine-tuning | DistilBERT | 76.6% |
