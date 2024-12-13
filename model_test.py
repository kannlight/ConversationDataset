from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import torch
import json

# アノテーション用モデルのロード
tokenizer = AutoTokenizer.from_pretrained("Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
config = LukeConfig.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)
model = AutoModelForSequenceClassification.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)

# アノテーション対象のデータを読み込む
data = {}
with open('data/2xsyj5ipp7xivr3b3456aurn.json', 'r') as f:
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

print(texts)
max_seq_length=256
token=tokenizer(texts,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt")
output=model(token['input_ids'], token['attention_mask'])
labels = output.logits.clone().detach().softmax(dim=1)

if 'data' in data:
    for i, talk in enumerate(data['data']):
        talk['label'] = labels[i].tolist()

with open('data/2xsyj5ipp7xivr3b3456aurn.json', 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

max_index=torch.argmax(output.logits.clone().detach(), dim=-1)

for i, max in enumerate(max_index):
    print(texts[i])
    if max==0:
        print('joy、うれしい')
    elif max==1:
        print('sadness、悲しい')
    elif max==2:
        print('anticipation、期待')
    elif max==3:
        print('surprise、驚き')
    elif max==4:
        print('anger、怒り')
    elif max==5:
        print('fear、恐れ')
    elif max==6:
        print('disgust、嫌悪')
    elif max==7:
        print('trust、信頼')