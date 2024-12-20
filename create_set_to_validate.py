# アノテーション精度の人手評価用に、設問となるjsonファイルを作成
import os
import json
import random

def create_set(num_talk):
    all_talk = []
    for filename in os.listdir('dataset2'):
        data = {}
        with open('dataset2/'+filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'data' in data:
            for talk in data['data']:
                all_talk.append(talk)
    random.shuffle(all_talk)
    set = all_talk[:num_talk]
    with open('set_to_validate.json', 'w', encoding='utf-8') as f:
        json.dump({'data':set}, f, indent=4,ensure_ascii=False)

if __name__ == "__main__":
    create_set(100)
