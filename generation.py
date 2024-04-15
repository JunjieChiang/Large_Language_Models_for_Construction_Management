# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
from http import HTTPStatus
import logging
import dashscope
import json
import info_retrieval

def integrate_docs_to_context(relevant_docs):
    """将相关文档内容整合为一段上下文文本"""
    context = "\n".join(json.dumps(doc, ensure_ascii=False) for doc in relevant_docs)

    return context

def call_with_messages(query, context):

    prompt = f"""
    你将接收到一些使用json字典格式存储的基础资料对象用于解答用户的询问。该对象是面向BIM模型中各个基础构建的JSON数据对象，这些对象具有以下键及其结构：
    name：构件名称，global_id：构件id，properties：一个包含额外属性的对象集。其中有一些构件属性会直接显示在构件名称中，你需要根据用户提到的构件名称到其相应的properties中进行相关信息的查询。
    '''{context}'''。
    此外，你接收到的用户问题是：'''{query}'''。
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

query = '单扇窗高度是多高？'
data_path = 'examples/BIM.json'
model_path = 'retriever/multilingual-e5-large'
index_path = 'result/faiss_index.index'
json_name_path = 'result/json_obj_name.json'
relevant_docs = info_retrieval.main(query, data_path, model_path, index_path, json_name_path)
context = integrate_docs_to_context(relevant_docs)
call_with_messages(query, context)

if __name__ == '__main__':

    print()
