from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import torch
import json
import os

# アノテーション用モデルのロード
tokenizer = AutoTokenizer.from_pretrained("Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
config = LukeConfig.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)
model = AutoModelForSequenceClassification.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)

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
    # 確率分布をラベルとする
    labels = output.logits.clone().detach().softmax(dim=1)

    # ラベルを付与
    if 'data' in data:
        for i, talk in enumerate(data['data']):
            talk['label'] = labels[i].tolist()

    # 保存
    with open('dataset/'+filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.remove('data_to_annotate/'+filename)

if __name__ == "__main__":
    for file in os.listdir('data_to_annotate'):
        annotate(file)
