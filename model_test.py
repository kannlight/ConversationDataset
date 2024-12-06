from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import torch
tokenizer = AutoTokenizer.from_pretrained("Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
config = LukeConfig.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)    
model = AutoModelForSequenceClassification.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)

# print(config)

text='😆'

max_seq_length=512
token=tokenizer(text,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt")
output=model(token['input_ids'], token['attention_mask'])
max_index=torch.argmax(torch.tensor(output.logits))
print(torch.tensor(output.logits))

if max_index==0:
    print('joy、うれしい')
elif max_index==1:
    print('sadness、悲しい')
elif max_index==2:
    print('anticipation、期待')
elif max_index==3:
    print('surprise、驚き')
elif max_index==4:
    print('anger、怒り')
elif max_index==5:
    print('fear、恐れ')
elif max_index==6:
    print('disgust、嫌悪')
elif max_index==7:
    print('trust、信頼')