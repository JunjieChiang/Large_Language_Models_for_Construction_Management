import json
import logging
import numpy as np
import faiss
import argparse
from FlagEmbedding import FlagModel
import vectorized_KB


def setup_logging():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def search_index(index_file, model, query, data, top_k):
    index = faiss.read_index(index_file)
    query_vector = np.array([model.encode(query)]).astype('float32')

    if index is not None:
        D, I = index.search(query_vector, top_k)
        # matched_names = [names[idx] for idx in I[0]]
        # logging.info(f"Matched names:\n {matched_names}")

        matched_items_properties = [data[idx]['properties'] for idx in I[0]]

        return matched_items_properties

    else:
        logging.error("FAISS index is not loaded.")

        return []


def load_resources(data_file):

    with open(data_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    # logging.info(f"JSON {data_file} Uploaded")

    return data


def main(query, data_file, index_file, embedding_model, top_k):
    # setup_logging()

    # 创建模型实例（确保模型实例可以重用，或者仅在需要时创建）
    model = vectorized_KB.load_embedding_model(embedding_model)

    # 加载预先计算的资源
    data = load_resources(data_file)

    # 执行搜索
    relevant_docs = search_index(index_file, model, query, data, top_k)

    return relevant_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve information from precomputed BIM model data.")
    parser.add_argument("query", help="Query string to process")
    args = parser.parse_args()

    index_path = 'result/faiss_index.index'
    embeddings_path = 'result/embeddings.json'
    names_path = 'result/names.json'

    # relevant_docs = main(args.query, index_path, embeddings_path, names_path)
