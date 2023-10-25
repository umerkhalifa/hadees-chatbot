import nltk
import os
import evaluate
import numpy as np
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import time 
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
nltk.download("punkt", quiet=True)


class DataProcessing:
    def __init__(self, csv_path, columns, model_name):
        self.path = csv_path
        self.columns = columns
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

    # preprocessing function
    def preprocess_function(self,examples):
        """Add prefix to the sentences, tokenize the text, and set the labels"""
        prefix = "answer the question: "
        # The "inputs" are the tokenized answer:
        inputs = [prefix + doc for doc in examples["question"]]
        model_inputs = self.tokenizer(inputs, max_length=128, truncation=True)
        
        
        # The "labels" are the tokenized outputs:
        labels = self.tokenizer(text_target=examples["answer"], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def load_custom_dataset(self, train_ratio=0.7, test_ratio=0.2, validation_ratio=0.1):
        # Load your CSV file into a Pandas DataFrame
        data_df = pd.read_csv(self.path, usecols = self.columns)

        # Split the DataFrame into train, test, and validation DataFrames
        train_df, remaining_df = train_test_split(data_df, train_size=train_ratio, random_state=42)
        test_df, validation_df = train_test_split(remaining_df, test_size=test_ratio / (test_ratio + validation_ratio), random_state=42)

        # Convert DataFrames to Dataset objects
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        validation_dataset = Dataset.from_pandas(validation_df)

        # Create a DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
            "validation": validation_dataset
        })
        
        tokenized_dataset = dataset_dict.map(self.preprocess_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['question', 'answer', '__index_level_0__','attention_mask',])
        return tokenized_dataset
    
class FineTune:
    def __init__(self, model_name, tokenized_dataset):
        self.model_name = model_name
        self.tokenized_dataset = tokenized_dataset
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
    def compute_metrics(self,eval_preds):
        preds, labels = eval_preds

        # decode preds and labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        metric = evaluate.load("rouge")
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return result   
        
    def train_evaluate(self):
        # Load the tokenizer, model, and data collator
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model) 
        
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        print(f"""trainable model parameters: {trainable_model_params}\n
        all model parameters: {all_model_params}\n
        percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%""")

        output_dir = f'./hadees-qa-training-{str(int(time.time()))}'
        
        # Set up training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir= output_dir,
            evaluation_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
            push_to_hub=False
        )

        # Set up trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        trainer.train()   

    
if __name__ == '__main__':
    # Model to finetune
    model_name = "google/flan-t5-base"

    # Loading and Preprocessing dataset
    path = './hadees_qa.csv'
    columns = ['question', 'answer']
    preprocess = DataProcessing(path, columns, model_name)
    tokenized_dataset = preprocess.load_custom_dataset()

    # Finetuning
    model = FineTune(model_name, tokenized_dataset)
    model.train_evaluate()