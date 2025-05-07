from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 사용할 모델 이름
MODEL_NAME = "klue/bert-base"

# 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# 로컬 model/ 폴더에 저장
tokenizer.save_pretrained("model")
model.save_pretrained("model")

print("✅ 모델 저장 완료!")
