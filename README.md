# hadees-chatbot
The idea of the project is to finetune a llm to answer islamic questions related questions
![Example](https://github.com/umerkhalifa/hadees-chatbot/blob/main/bot.gif)

# Dataset and Model
1. The questions and answers from https://islamqa.info/en was scraped and preprocessed.
2. The model = "google/flan-t5-base" was used for finetuning

# Chat bot
Streamlit was used to build chat board for the finetuned model.

# Note
Please note that the model is not fully optimized therefore it needs toxicity check before it is used.
