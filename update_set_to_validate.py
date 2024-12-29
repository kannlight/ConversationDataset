# アノテーション精度の人手評価用に、設問となるjsonファイルを作成(新たなモデルでアノテーションしたデータセットから、同じ対話に関するjsonを作り直す)
import os
import json
import random

def create_set(file):
    vali_set = {}
    with open(file, 'r', encoding='utf-8') as f:
        vali_set = json.load(f)
    vali_uris = []
    if 'data' in vali_set:
        for vali_talk in vali_set['data']:
            vali_uris.append(vali_talk['uri'])

    all_talk = []
    for filename in os.listdir('dataset3'):
        data = {}
        with open('dataset3/'+filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'data' in data:
            for talk in data['data']:
                all_talk.append(talk)
    set = []
    for uri in vali_uris:
        for talk in all_talk:
            if talk['uri'] == uri:
                set.append(talk)

    with open('set_to_validate2.json', 'w', encoding='utf-8') as f:
        json.dump({'data':set}, f, indent=4,ensure_ascii=False)

if __name__ == "__main__":
    json_file = "set_to_validate.json"
    create_set(json_file)
