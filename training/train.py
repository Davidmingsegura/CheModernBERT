import os
import torch
import pandas as pd
import numpy as np
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback
import wandb
import time
from transformers import TrainerCallback

class StepTimer(TrainerCallback):
    def __init__(self):
        self.step_times = []

    def on_step_begin(self, args, state, control, **kwargs):

        if state.is_world_process_zero:
            self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.step_times.append(time.perf_counter() - self._t0)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.step_times:
            avg = sum(self.step_times) / len(self.step_times)
            print(f"\n=== Average wall-time/step on {args.world_size} GPUs: {avg:.4f}s ===\n")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

batch_size_per_gpu = 8
batch_size = batch_size_per_gpu  

model_path = "answerdotai/ModernBERT-base"
# new_tokens = [
#      "[START_MOL]", "[END_MOL]", "[START_SMILES]", "[END_SMILES]", "[URL]", "[START_FIGURE]", "[END_FIGURE]", "[START_TABLE]", "[END_TABLE]", "[START_BIBREF]", "[END_BIBREF]", "[START_FORMULA]", "[END_FORMULA]"
# ]

tokenizer = AutoTokenizer.from_pretrained(model_path)
# num_added_tokens = tokenizer.add_tokens(new_tokens)
# print(f"Added {num_added_tokens} tokens: {new_tokens}")

model = AutoModelForMaskedLM.from_pretrained(model_path, reference_compile=False).to(device)
# model.resize_token_embeddings(len(tokenizer))
model.config.output_hidden_states = True

for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

model.eval()

orig_file_path = "/iopsstor/scratch/cscs/davidsegura/dataset/dataset_8192_tok_wo_tags_wo_comp_profile.csv"
df_orig = pd.read_csv(orig_file_path, engine='python', on_bad_lines='skip')
print("Original dataset shape:", df_orig.shape)

new_dataset_path = "/iopsstor/scratch/cscs/davidsegura/dataset/procedure_processed_files_wo_tags.csv"
df_new = pd.read_csv(new_dataset_path, engine='python', on_bad_lines='skip')
print("New dataset shape:", df_new.shape)

print(f"datasets used for this run: {orig_file_path},{new_dataset_path}. Full 1M5 of the no tag procedures")

text_list_orig = df_orig['text_clean'].dropna().tolist()
text_list_new = df_new['processed_text_clean'].dropna().tolist()

combined_texts = text_list_orig + text_list_new
print(f"Total number of prompt texts before sampling: {len(combined_texts)}")

np.random.shuffle(combined_texts)

print(f"Total number of prompt texts after sampling: {len(combined_texts)}")

train_texts, val_texts = train_test_split(combined_texts, test_size=0.2, random_state=42)
print(f"Training samples: {len(train_texts)}; Validation samples: {len(val_texts)}")

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})


def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=8192, return_tensors="pt")

    print(f"Original batch size: {len(examples['text'])}")  
    print(f"First tokenized sequence length: {len(tokens['input_ids'][0])}") 
    print(f"Tokenized batch shape: {len(tokens['input_ids'])}")  

    return tokens  

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

if local_rank == 0:
    wandb.init(
        project="cpt-modernbert",
        name="8192-tok-mlm-new-procedures-60-epochs",
        id="u1a3mw8b",    
        resume="must"     
    )

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    pad_to_multiple_of=8192
)


training_args = TrainingArguments(
    output_dir="/iopsstor/scratch/cscs/davidsegura/scripts/results",
    num_train_epochs=60,               
    per_device_train_batch_size=batch_size_per_gpu,
    save_strategy="epoch",            
    save_total_limit=2,     
    evaluation_strategy="epoch",       
    logging_steps=1000,
    bf16=True,
    report_to="wandb",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[StepTimer()], 
)


print("All processes: about to start training.")
# trainer.train()
last_checkpoint = "/iopsstor/scratch/cscs/davidsegura/scripts/results/checkpoint-521136"

trainer.train(resume_from_checkpoint=last_checkpoint)

print("Model trained!")

if local_rank == 0:
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    path = '/iopsstor/scratch/cscs/davidsegura/models/fine-web-modernbert-base-8192-multi-tok-new-procedure-60-epochs-notags'
    model_to_save.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model and tokenizer saved!, the model is called {path}")

torch.distributed.barrier()

def get_cls_embeddings(texts, batch_size=1):
    model.eval()  
    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokenized_inputs = tokenizer(batch, padding="max_length", truncation=True, return_tensors="pt", max_length=8192)
            tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
            total_tokens = sum(len(seq) for seq in tokenized_inputs["input_ids"])
            print("Sum of tokenized sequence lengths in batch:", total_tokens)

            print(f"Batch {i // batch_size}: Input shape:", tokenized_inputs["input_ids"].shape)  

            outputs = model(**tokenized_inputs, output_hidden_states=True)

            for idx, hidden_state in enumerate(outputs.hidden_states):
                print(f"Layer {idx} output shape:", hidden_state.shape)

            last_hidden = outputs.hidden_states[-1]
            print("Final hidden state shape:", last_hidden.shape)

            if last_hidden.dim() == 3:
                embeddings = last_hidden[:, 0, :]
            else:
                print("Hidden state has unexpected shape:", last_hidden.shape)
                embeddings = last_hidden  

            embeddings_list.append(embeddings.cpu().numpy())

    return np.concatenate(embeddings_list, axis=0)

model_path = "/iopsstor/scratch/cscs/davidsegura/models/fine-web-modernbert-base-8192-multi-tok-new-procedure-60-epochs-notags"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
