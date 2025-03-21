import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# تحميل البيانات من ملف CSV
dataset = load_dataset("csv", data_files="/mnt/data/couples_chat_arabic_large.csv")

# استخدام موديل DialoGPT أو أي موديل متاح للغة العربية
model_name = "facebook/opt-1.3b"  # يمكنك تجربة "meta-llama/Llama-2-7b-chat-hf" إذا كان لديك موارد قوية
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# تحضير البيانات: تحويل المحادثات إلى Tokens
def tokenize_function(examples):
    return tokenizer(examples["message"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# تقسيم البيانات إلى تدريب واختبار
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# ضبط عملية التدريب
training_args = TrainingArguments(
    output_dir="./chat_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    push_to_hub=False,
    report_to="none"
)

# تجهيز Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # نحن ندرب نموذج محادثة، لذلك لا نريد Masked LM
)

# تدريب النموذج
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# حفظ النموذج بعد التدريب
model.save_pretrained("./trained_chat_model")
tokenizer.save_pretrained("./trained_chat_model")
