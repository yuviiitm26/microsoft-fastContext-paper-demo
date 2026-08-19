#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# In[2]:


!pip install --upgrade torchao

# In[3]:


!pip install -q --upgrade torchao peft transformers datasets accelerate

# In[4]:


# =====================================================================
# 1. SETUP & IMPORTS
# =====================================================================
# !pip install -q transformers datasets peft evaluate accelerate scikit-learn

import os
import gc
import re
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# Ensure strict reproducibility across runs
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)

# =====================================================================
# 2. COMPETITION CONFIGURATION
# =====================================================================
class Config:
    # Using base model for high speed (3x faster) while maintaining high accuracy
    MODEL_NAME = "microsoft/deberta-v3-base" 
    MAX_LENGTH = 512
    NUM_LABELS = 30
    BATCH_SIZE = 16          # High batch size for speed on T4
    EPOCHS = 5               # Optimal early stopping range
    LR = 1e-4                
    N_SPLITS = 3             # 3-Fold Ensemble (Best balance of time/performance)
    SEED = 42

# =====================================================================
# 3. ELITE PREPROCESSING & TOKENIZATION
# =====================================================================
print("Loading data...")
# Adjust file paths based on your Kaggle input directory (e.g., '/kaggle/input/nppe-challenge/train.csv')
train_df = pd.read_csv("/kaggle/input/competitions/dlp-nppe-1-t-22026/train.csv")
test_df = pd.read_csv("/kaggle/input/competitions/dlp-nppe-1-t-22026/test.csv")

# Initialize the fast C-backed tokenizer
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)

def clean_legal_text(text):
    """Maximizes 'Information Density' by removing noise that wastes token space."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\*+', '', text)                     # Remove decorative asterisks
    text = re.sub(r'http[s]?://\S+', '', text)          # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)                 # Remove Emails
    text = re.sub(r'\n+', ' ', text)                    # Flatten newlines (crucial token saver)
    text = re.sub(r'\s{2,}', ' ', text)                 # Flatten excessive spaces
    text = re.sub(r'[^A-Za-z0-9\s.,;:\-\'\"()%/]', '', text) # Keep only standard alphanumeric + basic punctuation
    return text.strip()

def head_middle_tail_tokenize(text, max_len=512):
    """Extracts 128 (Head) + 128 (Middle) + 254 (Tail) tokens."""
    clean_text = clean_legal_text(text)
    tokens = tokenizer.encode(clean_text, add_special_tokens=False)
    
    if len(tokens) <= max_len - 2:
        return tokenizer(clean_text, truncation=True, max_length=max_len)
    
    head_tokens = tokens[:128]               
    middle_start = len(tokens) // 2 - 64
    middle_tokens = tokens[middle_start : middle_start + 128] 
    tail_tokens = tokens[-254:]              
    
    input_ids = [tokenizer.cls_token_id] + head_tokens + middle_tokens + tail_tokens + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def preprocess_pipeline(examples):
    tokenized = [head_middle_tail_tokenize(text, Config.MAX_LENGTH) for text in examples["text"]]
    return {
        "input_ids": [t["input_ids"] for t in tokenized],
        "attention_mask": [t["attention_mask"] for t in tokenized]
    }

print("Running advanced cleaning and tokenization pipeline...")
full_train_ds = Dataset.from_pandas(train_df[['text', 'label']])
test_ds = Dataset.from_pandas(test_df[['id', 'text']])

# Utilize multi-core processing to tokenize instantly
full_train_ds = full_train_ds.map(preprocess_pipeline, batched=True, remove_columns=["text"], num_proc=4)
test_ds = test_ds.map(preprocess_pipeline, batched=True, remove_columns=["text"], num_proc=4)

# =====================================================================
# 4. ENSEMBLE TRAINING LOOP (LoRA)
# =====================================================================
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, preds)}

test_logits_accumulator = np.zeros((len(test_df), Config.NUM_LABELS))
skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
    print(f"\n" + "="*50)
    print(f"🚀 STARTING FOLD {fold_idx + 1} / {Config.N_SPLITS}")
    print("="*50)
    
    fold_train_ds = full_train_ds.select(train_idx)
    fold_val_ds = full_train_ds.select(val_idx)
    
    # Initialize base model freshly for each fold
    base_model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=Config.NUM_LABELS
    )
    
    # Comprehensive LoRA Configuration targeting all dense matrices
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=32,                         
        lora_alpha=64, 
        lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj", "output_proj", "dense", "classifier"]
    )
    model = get_peft_model(base_model, peft_config)
    
    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold_idx}",
        learning_rate=Config.LR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        num_train_epochs=Config.EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        label_smoothing_factor=0.05,  # Prevents overconfidence across the 30 classes
        fp16=False,
        bf16=False,                    # BFloat16 mixed precision to avoid DeBERTa instability
        warmup_ratio=0.1,             
        lr_scheduler_type="cosine",   
        report_to="none",             # Disables WandB/Tensorboard logging overhead
        logging_steps=100
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=fold_train_ds,
        eval_dataset=fold_val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Failsafe against overfitting
    )
    
    trainer.train()
    
    print(f"Generating test predictions for Fold {fold_idx + 1}...")
    fold_predictions = trainer.predict(test_ds)
    test_logits_accumulator += fold_predictions.predictions
    
    # Force aggressive VRAM cleanup before starting the next fold
    del base_model, model, trainer
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================================
# 5. SUBMISSION GENERATION (HOST TYPO OVERRIDE)
# =====================================================================
print("\n" + "="*50)
print("Blending fold outputs and saving final submission...")
# Soft-Voting: Selecting the class with the highest combined logit scores across all folds
final_preds = np.argmax(test_logits_accumulator, axis=1)

submission = pd.read_csv("/kaggle/input/competitions/dlp-nppe-1-t-22026/sample_submission.csv")
submission['label'] = final_preds

# OVERRIDE: Forces the column name to match the evaluator's hidden uppercase expectation
submission.columns = ['ID', 'label']

submission.to_csv("submission.csv", index=False)
print("🏆 Notebook Execution Complete. Submission file generated as 'submission.csv'!")

# In[ ]:




# In[ ]:



