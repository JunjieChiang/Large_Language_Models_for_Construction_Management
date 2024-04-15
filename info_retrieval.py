import json
import logging
import numpy as np
import faiss
from FlagEmbedding import FlagModel
import vectorized_KB


def setup_logging():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_faiss_index(index_file):

    try:
        index = faiss.read_index(index_file)
        logging.info(f"FAISS index loaded with {index.ntotal} vectors.")

        return index

    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}")

        return None

def load_names(json_name_path):

    with open(json_name_path, 'r', encoding='utf-8') as file:
        names = json.load(file)
    logging.info(f"Json object name loaded: {json_name_path}")

    return names


def encode_query(model, query):

    query_vector = model.encode(query)

    return np.array([query_vector]).astype('float32')


def search_index(index, names, query, query_vector, data, k=1):

    if index is not None:
        D, I = index.search(query_vector, k)
        logging.info(f"User query: {query}")
        logging.info(f"Top {k} most similar items indices: {I}")
        logging.info(f"Corresponding distances: {D}")

        matched_names = [names[idx] for idx in I[0]]        # 获取匹配的"name"
        logging.info(f"Matched names:\n {matched_names}")

        matched_items = []
        for category in data.values():  # 迭代字典的值（每个类别的列表）
            for item in category:  # 迭代每个类别列表中的项目
                if item['name'] in matched_names:
                    matched_items.append(item)
        logging.info(f"Matched Json objects:\n {matched_items}")

        return matched_names, matched_items

    else:

        logging.error("FAISS index is not loaded.")

        return [], []

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


if __name__ == "__main__":
    query = '请问砖墙的厚度有多厚？'
    data_path = 'examples/BIM.json'
    model_path = 'retriever/multilingual-e5-large'
    index_path = 'result/faiss_index.index'
    json_name_path = 'result/json_obj_name.json'

    main(query, data_path, model_path, index_path, json_name_path)

