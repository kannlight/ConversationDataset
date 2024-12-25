from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import torch
import json
import os

# アノテーション用モデルのロード
tokenizer = AutoTokenizer.from_pretrained("patrickramos/bert-base-japanese-v2-wrime-fine-tune")
model = AutoModelForSequenceClassification.from_pretrained('patrickramos/bert-base-japanese-v2-wrime-fine-tune')

def annotate(filename):
    # アノテーション対象のデータを読み込む
    data = {}
    with open('data_to_annotate/'+filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 各対話から最終発話をtextに追加（最終発話が複数続いたら改行で繋げる）
    texts = []
    if 'data' in data:
        for talk in data['data']:
            t = []
            for utter in talk['talk']:
                if utter['type'] == 3:
                    t.append(utter['utter'])
            texts.append('\n'.join(t))

    # トークン化
    max_seq_length=256
    token=tokenizer(texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt")
    
    # モデルに推定させる
    output=model(token['input_ids'], token['attention_mask'])
    # マルチラベル分類用
    reader_score = output.logits.clone().detach()[:,-8:]
    # シグモイド関数で確率に変換
    labels = reader_score.sigmoid()

    # ラベルを付与
    if 'data' in data:
        for i, talk in enumerate(data['data']):
            talk['label'] = labels[i].tolist()

    # 保存
    with open('dataset3/'+filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.remove('data_to_annotate/'+filename)

if __name__ == "__main__":
    for file in os.listdir('data_to_annotate'):
        annotate(file)
