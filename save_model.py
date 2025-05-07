from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "klue/bert-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained("model")
model.save_pretrained("model")

print("✅ 모델 저장 완료!")
