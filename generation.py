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
    你将接收到一些面向BIM模型中各基础构件的JSON数据对象，这些对象具有以下键及其结构：name：构件名称，global_id：构件id，properties：该构件的属性集，该属性集及其相应解释如下：
        "IsExternal": 该构件是否在外部？,
        "LoadBearing": 该构件是否是承重结构构件？,
        "ExtendToStructure": 该构件是否延伸到了结构?,
        "Slope": 该构件的倾斜程度,
        "Reference": 该构件的型号,
        "Span": 该构件的间隔范围,
        "类别": 该构件的类别,
        "族": 该构件的族,
        "族与类型": 该构件的族与类型,
        "类型": 该构件类型,
        "类型 ID": 该构件类型 ID,
        "连接状态": 该构件的连接状态,
        "体积": 该构件的体积,
        "底部高程": 该构件的底部高程,
        "长度": 该构件的长度,
        "顶部高程": 该构件的顶部高程,
        "结构材质": 该构件的结构材质,
        "部件名称": 该构件的部件名称,
        "参照标高": 该构件的参照标高,
        "参照标高高程": 该构件的参照标高高程,
        "横截面旋转": 该构件的横截面旋转,
        "终点标高偏移": 该构件的终点标高偏移,
        "起点标高偏移": 该构件的起点标高偏移,
        "结构用途": 该构件的结构用途,
        "剪切长度": 该构件的剪切长度,
        "创建的阶段": 该构件的创建阶段,
        "OverallHeight": 该构件的高度,
        "OverallWeight": 该构件的宽度
    检索到的JSON对象中包含了name, global_id和properties，properties中有键和值，键指的是属性名称，值则是具体的属性值。你需要根据上面对属性集“键”的解释以及以下检索到的JSON对象中相应“键“的“值”来回答用户的查询需求。
    检索到的相关JSON对象如下：
    '''{context}'''
    你接收到的用户问题是：
    '''{query}'''
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
parser.add_argument('--embedding_model', type=str, default='retriever\multilingual-e5-large',
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

    relevant_docs = info_retrieval.main(args.query, args.data_path, args.index_path, args.vectorised_KB, args.json_name_path, args.top_k)
    context = integrate_docs_to_context(relevant_docs)
    call_with_messages(args.query, context)
