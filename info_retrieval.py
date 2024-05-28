import json
import logging
import numpy as np
import faiss
import argparse
from FlagEmbedding import FlagModel
import vectorized_KB


def setup_logging():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def load_faiss_index(index_file):

    try:
        index = faiss.read_index(index_file)
        # logging.info(f"FAISS index loaded with {index.ntotal} vectors.")

        return index

    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}")

        return None

def load_names(json_name_path):

    with open(json_name_path, 'r', encoding='utf-8') as file:
        names = json.load(file)
    # logging.info(f"Json object name loaded: {json_name_path}")

    return names


def encode_query(model, query):

    query_vector = model.encode(query)

    return np.array([query_vector]).astype('float32')


def search_index(index, names, query, query_vector, data, top_k):

    if index is not None:
        D, I = index.search(query_vector, top_k)
        # logging.info(f"User query: {query}")
        # logging.info(f"Top {top_k} most similar items indices: {I}")
        # logging.info(f"Corresponding distances: {D}")

        matched_names = [names[idx] for idx in I[0]]        # 获取匹配的"name"
        # logging.info(f"Matched names:\n {matched_names}")

        matched_items = []
        for category in data.values():  # 迭代字典的值（每个类别的列表）
            for item in category:  # 迭代每个类别列表中的项目
                if item['name'] in matched_names:
                    matched_items.append(item)
        # logging.info(f"Matched Json objects:\n {matched_items}")

        return matched_names, matched_items

    else:

        logging.error("FAISS index is not loaded.")

        return [], []

'''
def main(query, data_path, model_path, index_path, json_name_path):

    setup_logging()

    # 召回相关数据
    data = vectorized_KB.load_data(data_path)
    model = vectorized_KB.load_embedding_model(model_path)
    query_vector = encode_query(model, query)
    index = load_faiss_index(index_path)
    names = load_names(json_name_path)
    matched_names, matched_items = search_index(index, names, query, query_vector, data)
    relevant_docs = [item['properties'] for item in matched_items]
    # print(relevant_docs)

    return relevant_docs
'''

def load_resources(data_file, index_file, embeddings_file, names_file):

    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # logging.info(f"JSON {data_file} Uploaded")

    with open(embeddings_file, 'r', encoding='utf-8') as file:
        embeddings = json.load(file)  # 加载已保存的embeddings
    # logging.info(f"Embeddings uploaded.")

    index = faiss.read_index(index_file)    # 加载已保存的FAISS索引
    # logging.info(f"FAISS index loaded with {index.ntotal} vectors.")

    with open(names_file, 'r', encoding='utf-8') as file:
        names = json.load(file)            # 加载已保存的names
    # logging.info(f"JSON name loaded")

    return data, index, embeddings, names


def main(query, data_file, index_file, embedding_model, embedding_file, names_file, top_k):
    setup_logging()

    # 创建模型实例（确保模型实例可以重用，或者仅在需要时创建）
    model = vectorized_KB.load_embedding_model(embedding_model)

    # 加载预先计算的资源
    data, index, embeddings, names = load_resources(data_file, index_file, embedding_file, names_file)

    # logging.info("ALL precomputed resources loaded successfully.")

    # 向量化查询
    query_vector = encode_query(model, query)

    # 执行搜索
    matched_names, matched_items = search_index(index, names, query, query_vector, data, top_k)

    # 处理匹配到的文档
    relevant_docs = [item['properties'] for item in matched_items]

    return relevant_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve information from precomputed BIM model data.")
    parser.add_argument("query", help="Query string to process")
    args = parser.parse_args()

    index_path = 'result/faiss_index.index'
    embeddings_path = 'result/embeddings.json'
    names_path = 'result/names.json'

    # relevant_docs = main(args.query, index_path, embeddings_path, names_path)

