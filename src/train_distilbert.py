from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import(
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score,f1_score
base_dir=Path(__file__).resolve().parent.parent
data_dir=base_dir/"data"
model_dir=base_dir/"models"
report_dir=base_dir/"reports"
model_dir.mkdir(exist_ok=True)
report_dir.mkdir(exist_ok=True)
model_name="distilbert/distilbert-base-uncased"

def train_task(target_column):
    print(f"\nFine-tuning DistilBERT for {target_column}")
    train_df=pd.read_csv(data_dir/"train.csv")
    val_df=pd.read_csv(data_dir/"validation.csv")
    test_df=pd.read_csv(data_dir/"test.csv")
    labels=sorted(train_df[target_column].unique())
    label_to_id={
        label:index for index,label in enumerate(labels)
    }
    id_to_label={
        index:label for label,index in label_to_id.items()
    }
    for frame in [train_df,val_df,test_df]:
        unknown=set(frame[target_column])-set(labels)
        if unknown:
            raise ValueError(f"Unknown labels:{unknown}")
        frame["label"]=frame[target_column].map(label_to_id)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    def prepare_dataset(frame):
        dataset=Dataset.from_pandas(
            frame[["text","label"]].reset_index(drop=True)
        )
        dataset=dataset.map(tokenize_batch,batched=True)
        dataset=dataset.remove_columns(["text"])
        dataset.set_format("torch")
        return dataset
    train_dataset=prepare_dataset(train_df)
    val_dataset=prepare_dataset(val_df)
    test_dataset=prepare_dataset(test_df)
    model=AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id
    )
    def calculate_metrics(eval_result):
        logits,true_labels=eval_result
        predictions=np.argmax(logits,axis=1)
        return{
            "accuracy":accuracy_score(true_labels,predictions),
            "macro_f1":f1_score(
                true_labels,
                predictions,
                average="macro",
                zero_division=0
            )
        }
    output_dir=model_dir/f"distilbert_{target_column}"
    arguments=TrainingArguments(
        output_dir=str(output_dir/"checkpoint"),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=42
    )
    trainer=Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=calculate_metrics
    )
    start_time=time.perf_counter()
    trainer.train()
    training_seconds=time.perf_counter()-start_time
    validation_metrics =trainer.evaluate(val_dataset)
    start_time=time.perf_counter()
    test_metrics=trainer.evaluate(test_dataset)
    test_inference_seconds=(time.perf_counter()-start_time)
    
    output_dir.mkdir(parents=True,exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    report={
        "task":target_column,
        "labels":labels,
        "training_seconds":round(training_seconds,3),
        "validation":{
            key:float(value)
            for key,value in validation_metrics.items()
            if key.startswith("eval_")
        },
        "test":{
            key:float(value)
            for key,value in test_metrics.items()
           if key.startswith("eval_")
        },
        "test_evaluation_seconds":round(test_inference_seconds,3)
    }
    with open(report_dir/f"distilbert_{target_column}_results.json","w")as file:
        json.dump(report,file,indent=4)
    print(json.dump(report,indent=4))
if __name__=="__main__":
    train_task("category")
    train_task("priority")