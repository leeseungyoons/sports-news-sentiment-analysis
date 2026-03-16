import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


class SentimentAnalyzer:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        model_dir = base_dir / "model"

        self.use_fallback = False

        weight_candidates = [
            model_dir / "pytorch_model.bin",
            model_dir / "model.safetensors",
        ]

        if any(p.exists() for p in weight_candidates):
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            self.model.eval()
        else:
            print("[WARN] 모델 가중치가 없어 fallback 모드로 실행합니다.")
            self.use_fallback = True

            self.positive_words = [
                "승리", "대승", "역전승", "홈런", "활약", "맹타", "결승타", "극적인",
                "끝내기", "무실점", "이기며", "완봉", "연승", "호투", "기세",
                "눈부신", "역전극", "쾌조", "드라마틱", "MVP", "선발승",
                "기록 경신", "놀라운", "완벽한", "압도적"
            ]

            self.negative_words = [
                "패배", "병살타", "실책", "놓쳤다", "무득점", "패전", "무산", "부진",
                "역전패", "부상", "이탈", "불안", "혹사", "탈락", "결장", "퇴장",
                "공백", "논란", "낙하", "머리 부상", "병원", "사고", "사망",
                "실망", "안전 사고", "애도", "위험", "의심", "충격", "취소", "중단"
            ]

    def predict(self, text):
        if not self.use_fallback:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).squeeze()

            positive_score = probs[1].item()
            negative_score = probs[0].item()

            label = "Positive" if positive_score > negative_score else "Negative"
            return label, positive_score, negative_score

        pos_hits = sum(word in text for word in self.positive_words)
        neg_hits = sum(word in text for word in self.negative_words)

        if pos_hits == 0 and neg_hits == 0:
            positive_score = 0.50
            negative_score = 0.50
            label = "Negative"
        else:
            total = pos_hits + neg_hits
            positive_score = pos_hits / total
            negative_score = neg_hits / total
            label = "Positive" if positive_score > negative_score else "Negative"

        return label, float(positive_score), float(negative_score)