# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
from http import HTTPStatus
import logging
import dashscope
import json
import argparse
import info_retrieval

def integrate_docs_to_context(relevant_docs):
    """将相关文档内容整合为一段上下文文本"""
    context = "\n".join(json.dumps(doc, ensure_ascii=False) for doc in relevant_docs)

    return context

def call_with_messages(query, context):

    prompt = f"""
    你是处理JSON格式对象的专业人士。
    你将接收用户的查询问题：{query}
    你将接收数个JSON数据对象组成的字符串：{context}
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
        "Reference": 该构件的型号,
    
    你的任务是，从每个JSON对象中找到与用户查询最匹配的键值对，并回答用户问题；如输入的JSON对象中不存在对应的键值，则你的回答应该是：我不知道。
    除以上内容外，不要输出额外信息。
        """

    messages = [{'role': 'system', 'content': 'You are a professional assistant for construction managers.'},
                {'role': 'user', 'content': prompt}]

    # print(messages)
    response = dashscope.Generation.call(
        model='qwen1.5-72b-chat',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        content = response['output']['choices'][0]['message']['content']
        logging.info(f"Model Generation:\n{content}")
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


parser = argparse.ArgumentParser(description="Retrieve information and generate answers for BIM model queries.")
parser.add_argument('--query', type=str, default='嵌入到内部的基本墙有多厚？',
                    help="The user query to process.")
parser.add_argument('--embedding_model', type=str, default='retriever/bge-m3',
                    help="Retrieve relevant information, the model can be replaced by model name (directory: retriever)")
parser.add_argument('--data_path', type=str, default='examples/BIM.json',
                    help='original data source need to search')
parser.add_argument('--index_path', type=str, default='result/faiss_index.index',
                    help='speed up the searching process')
parser.add_argument('--vectorised_KB', type=str, default='result/embeddings.json',
                    help='embedded external knowledge')
parser.add_argument('--json_name_path', type=str, default='result/json_obj_name.json',
                    help='searching information according retrieved name')
parser.add_argument('--top_k', type=int, default=5,
                    help='the number of returned information chunks')

args = parser.parse_args()


if __name__ == '__main__':

    relevant_docs = info_retrieval.main(args.query, args.data_path, args.index_path, args.embedding_model, args.vectorised_KB, args.json_name_path, args.top_k)
    context = integrate_docs_to_context(relevant_docs)
    call_with_messages(args.query, context)
