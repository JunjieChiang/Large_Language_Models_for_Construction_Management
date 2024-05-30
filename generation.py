# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
import argparse
import info_retrieval
from models import *


parser = argparse.ArgumentParser(description="Retrieve information and generate answers for BIM model queries.")
parser.add_argument('--query', type=str, default='结构框架 : AR_FST-M-10_200外墙板的体积是多少？',
                    help="The user query to process.")
parser.add_argument('--generative_model', type=str, default='gpt-3.5-turbo', help='the generative model')
parser.add_argument('--embedding_model', type=str, default='retriever/bge-m3',
                    help="Retrieve relevant information, the model can be replaced by model name (directory: retriever)")
parser.add_argument('--data_path', type=str, default='examples/bim_kb.jsonl',
                    help='original data source need to search')
parser.add_argument('--index_path', type=str, default='result/faiss_index.index',
                    help='speed up the searching process')
parser.add_argument('--vectorised_KB', type=str, default='result/embeddings.json',
                    help='embedded external knowledge')
parser.add_argument('--json_name_path', type=str, default='result/json_obj_name.json',
                    help='searching information according retrieved name')
parser.add_argument('--top_k', type=int, default=1,
                    help='the number of returned information chunks')

args = parser.parse_args()


def integrate_docs_to_context(relevant_docs):
    """将相关文档内容整合为一段上下文文本"""
    context = "\n".join(json.dumps(doc, ensure_ascii=False) for doc in relevant_docs)

    return context


def call_with_messages(llm, query, context):
    prompt = f"""
    任务说明：
    根据用户的问题进行查询并生成响应，需要对检索出来的属性信息查询用户的问题。其中用户的问题如下：
    {query}
    其中与用户查询相关的属性信息如下,你需要从中提取用户查询的属性值:
    {context}
    
    你需要关注每个JSON对象的键（key）及其解释如下：
        "体积": 该构件的体积,
        "底部高程": 该构件的底部高程,
        "长度": 该构件的长度,
        "顶部高程": 该构件的顶部高程,
        "结构材质": 该构件的结构材质,
        "部件名称": 该构件的部件名称,
        "参照标高": 该构件的参照标高,
        "OverallHeight": 该构件的高度,
        "OverallWeight": 该构件的宽度,
        "Reference": 该构件的型号

    注意：
    - 如果属性存在，则生成响应；如果属性不存在，则回答该建筑实体不存在该构件或属性。
    - 除了回答查询问题外，不要输出额外信息
    """

    response = llm.get_completion(prompt)

    print(response)


def main():

    model_configs_path = f"model_configs/{args.generative_model}.json"

    relevant_docs = info_retrieval.main(args.query, args.data_path, args.index_path, args.embedding_model, args.top_k)

    context = integrate_docs_to_context(relevant_docs)

    llm = init_model_config(model_configs_path)

    call_with_messages(llm, args.query, context)


if __name__ == '__main__':

    main()