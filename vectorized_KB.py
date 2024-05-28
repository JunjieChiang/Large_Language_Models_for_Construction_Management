import json
import logging
import numpy as np
import faiss
import generation
from tqdm import tqdm
from FlagEmbedding import FlagModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    logging.info(f"JSON {filepath} Uploaded")
    return data

# multilingual-e5-large是1024维度
def load_embedding_model(model_path):
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="为这个JSON数据生成表示以用于检索相关属性：",
                      use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    # logging.info(f"Model {model_path} Uploaded")
    return model


def encode_names(model, data, json_name_path):
    '''
    将检索分为两个阶段：向量化JSON对象中的name，再根据query信息检索matched对象中的数据项
    '''

    embeddings = []
    names = []  # 用于保存每个元素的name，以便后续检索

    for item_type in tqdm(data, desc="Initializing"):
        for item in tqdm(data[item_type], desc=f"Encoding {item_type}", leave=True):
            name = item.get('name', '')  # 只关注"name"字段
            embedding = model.encode(name)  # 直接对"name"进行向量化
            embeddings.append(embedding)
            names.append(name)  # 保存"name"到列表

    embeddings = np.array(embeddings).astype('float32')
    logging.info("All data encoded")

    with open(json_name_path, 'w') as f:
        json.dump(names, f)
    logging.info(f"JSON object name save to {json_name_path}")
    return embeddings, names

def create_faiss_index(embeddings, dimension=1024): # multilingual-e5-large为1024维度
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logging.info(f"FAISS index created with {index.ntotal} vectors.")
    return index

def save_faiss_index(index: object, filepath: object) -> object:
    faiss.write_index(index, filepath)
    logging.info(f"FAISS index saved to {filepath}")

def save_embeddings_to_file(embeddings, filepath):
    with open(filepath, 'w') as f:
        json.dump(embeddings.tolist(), f)
    logging.info(f"Embedding Saved to {filepath}")

if __name__ == "__main__":
    args = generation.parser.parse_args()
    data = load_data(args.data_path)
    model = load_embedding_model(args.embedding_model)
    embeddings, names = encode_names(model, data, args.json_name_path)       # 根据检索方式的不同替换嵌入函数
    index = create_faiss_index(embeddings)
    save_faiss_index(index, args.index_path)
    save_embeddings_to_file(embeddings, args.vectorised_KB)
