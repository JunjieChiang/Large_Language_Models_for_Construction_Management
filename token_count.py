import json
import jieba

def count_tokens_from_file(file_path):
    total_tokens = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            document = json.loads(line)
            # 分词query
            tokens = list(jieba.cut(document['query']))
            total_tokens += len(tokens)
            # 分词pos
            for sentence in document['pos']:
                tokens = list(jieba.cut(sentence))
                total_tokens += len(tokens)
            # 分词neg
            for sentence in document['neg']:
                tokens = list(jieba.cut(sentence))
                total_tokens += len(tokens)
    return total_tokens

# 文件路径
file_path = 'examples/finetune/gpt.txt'

# 计算tokens数量
token_count = count_tokens_from_file(file_path)
print(f"总共的Token数量为：{token_count}")
