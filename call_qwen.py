from http import HTTPStatus
import dashscope
import re
import json


def process_llm_history(config, paras):
    his_k = config['HISTORY_K']
    histories = paras['histories']

    his_info = ''
    if len(histories)>(his_k+1):
        his_info = histories[ :-his_k]
    '，'.join(his_info)
    
    return his_info


def eval_response(str_answer, config=None, answer_key=None):
    str_answer = re.sub(r'\s+', '', str_answer)        
    answer = str_answer.strip()
    print('\t' + answer + '\n')
    
    try:
        answer = json.loads(str_answer)
    except:
        print('\t regular parsing fails, try finding json marks...')
        try:
            matches = re.findall(r'```json\s*(.+?)\s*```', str_answer, re.DOTALL)
            if len(matches)==0:
                matches = re.findall(r'```json({.+?})```', str_answer, re.DOTALL)
                if len(matches)==0:
                    str_answer = str_answer.replace('```json', '').strip()
                    str_answer = str_answer.replace('```', '').strip()
                else:
                    str_answer = matches[0]
            else:
                str_answer = matches[0]
                
            try:
                answer = json.loads(str_answer)
            except:
                str_answer = call_qwen(paras={'task':'repair json', 'texts':str_answer, 'query':'' }, config=config)
                answer = json.loads(str_answer)
                print('\t repair json object successfully ')
            print('\t successfully find json marks and parse the response')
        except:
            print('\t no json object parsed, using original input answer')
    
    if not answer_key==None:
        answer = answer[answer_key]
    return answer
    

def call_with_messages(prompt, q_model='turbo', api_key=None):
    dashscope.api_key = api_key
    
    if q_model=='turbo':
        model = dashscope.Generation.Models.qwen_turbo
    elif q_model=='plus':
        model = dashscope.Generation.Models.qwen_plus
    elif q_model=='max':
        model = dashscope.Generation.Models.qwen_max
    else:
        raise('no qwen model specified')
    
    try:
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}]

        response = dashscope.Generation.call(
            model = model,  # 模型选择 qwen_max qwen_turbo qwen_plus
            messages = messages,
            max_tokens = 1500,   # 最大字数限制
            top_p = 0.1,  # 多样性设置
            repetition_penalty = 1.2,   # 重复惩罚设置
            temperature = 0.1,  # 随机性设置
            result_format = 'message'  # 结果格式
        )

        return response
    except Exception as e:
        print(f"call Qwen error: {e}")
        return None

        
def call_qwen(model='turbo', **kwargs):
    try:
        api_key = kwargs['config']['QWEN_APIKEY']
    except:
        print('Bad config inputs...')
        return None
        
    try:
        task = kwargs['paras']['task']
        query = kwargs['paras']['query']
        texts = kwargs['paras']['texts']
    except:
        print('Bad paras inputs...')
        return None
    
    try:
        model = kwargs['paras']['model']
    except:
        pass
    
    try:
        his_info = process_llm_history(kwargs['config'], kwargs['paras'])
    except:
        his_info = ''
        
    if task=='talk':
        add_req = kwargs['paras']['add_req']
        prompt = f"""
                你将接收到一些使用json字典格式存储的基础资料：'''{texts}'''。
                你接收到用户的问题：'''{query}'''。
                你可能还会接收到历史对话中，你曾给用户提供的相关答案：'''{his_info}'''。
                请你根据已有资料和历史记录（如有），回答用户问题。{add_req}
            """    
    
    elif task=='summary':
        try:
            limit = kwargs['paras']['limit']
            add_req = kwargs['paras']['add_req']
        except:
            limit = int(len(texts)*0.2)
            add_req = ''
        prompt = f"""
                你将接收一段文字材料：{texts}
                你需要总结以上文字材料主要内容，字数不超过{limit}字。{add_req}
                你的回答只包含提炼后的文本，不要返回其他多余信息。
            """
    
    elif task=='analyze-paras':
        prompt = f"""
                对下面这段输入的文本内容进行分析：'''{texts}'''
                检查文本内容是否可基于一些明显标识，可将内容分为若干段落，这些明显标识指各类数字类标识，比如“1、”和“2、”等。
                你的回答只能是json形式，键是"answer"，值是"true"或"false"。
                除以上json返回外，不要返回其他多余信息。
            """

    elif task=='rewrite-paras':
        topic = kwargs['paras']['topic']
        example = '''
                {
                    "paras":
                    [
                        { "段落1标题": "段落1内容" },
                        { "段落2标题": "段落2内容" },
                        ...
                    ]
                }
            '''
        prompt =  f"""
                对下面这段输入的文本内容进行分段重写：'''{texts}'''
                在重写过程中，你要做到以下几点：
                第一，检查文档中可用于分段的明显标识，比如“1、”和“2、”等数字，按这些标识将文本内容分段。
                第二，你的重写要符合当前的主题：{topic}，根据该主题可以对原文进行适当修改。
                第三，重写后的文本体量要与原文本相似，不能缩减。
                你的回答只能是json形式，其键为"paras"，值为一个列表，包含多个键值对；每个键值对代表一个段落，键为你提炼的段落标题，值为完整段落内容。
                除以上json返回外，不要返回其他多余信息。你的返回结果的示例如下（注意其中的标题和内容需要根据实际内容替换）：
            """
        prompt += example
        
    elif task=='para-judge':
        prompt = f"""
                你将受到一个文档的原始标题'''{query}'''
                你还将受到另一个待定标题'''{texts}'''
                你要判断：待定标题是和原始标题表达的相似意思（返回true），或者待定标题更适合作为原始标题的下级标题（返回false）
                你的回答只能是json形式，键为"answer"，值为"true"或"false"，不要返回其他多余信息。
            """
               
    elif task=='summary_rewrite':
        limit = kwargs['paras']['limit']
        example = '''
                {
                    abstract:...
                    new_text:...
                }        
            '''
        prompt = f"""
                你将接收一段文字材料：{texts}
                你需要执行以下步骤：
                1）将输入的文字材料总结提炼为一段精简的摘要abstract，字数不超过{limit}字。
                2）将原文字材料重写为通顺的文本new_text，主要保持原文的风格，不要写多余的内容。
                你的回答必须是JSON格式，其中键为abstract和new_text，值分别为你提炼的摘要和重写后的文字。返回数据结构如下所示：
            """
        prompt += example
    
    elif task=='reason':
        example = '''
            {
                match: 0
            }
        '''
        prompt = f"""
                你会收到一个列表，包含有多个主题，主题之间使用逗号','分隔，你还会收到一个查询意图。
                你需要从输入的主题中，推断哪个最符合输入的查询意图，并返回该主题在输入主题列表中的位置下标（位置从0开始计算）
                你的回答必须是JSON格式，键为"match"，值为最符合的主体的位置下标
                例如，输入主题列表为：'''通用技术规划调度技术电动汽车调度模型1,低碳新能源低碳交通新能源汽车政策,通用技术控制技术V2G车网协同控制,智慧电网政策,通用技术通用设备系统V2G系统架构'''
                查询意图为：'''电动汽车调度'''
                你应该发现下标为0的主题最符合要求，因此你的返回如下例所示：
                 {example}
                
                现在，根据以上指示和示例，进行推断：
                输入主题列表: '''{texts}'''
                查询意图：'''{query}'''
            """
        
    elif task=='parse-titles': # this one is NOT supported by API currently
        model = 'plus'
        example = '''一个输出例子如下：
                {
                  "ROOT": {
                    "1 发展现状": {
                      "1.1 市场情况": {},
                      "1.2 政策情况": {},
                      "1.3 技术情况": {}
                      },
                    
                    "2研究内容": {
                      "2.1 研究内容1": {},
                      "2.2 研究内容2": {}
                    }
                    
                    "技术现状": {}
                  }
                }
            '''
        instruct = f"""
                现在，请根据上述指令和示例，解析下面的输入标题：'''{texts}'''
            """
        prompt = f"""
                你会收到系列文档的标题
                你需要将输入的标题转化为JSON字典的树状结构，父节点是高层级标题，子叶节点是低层级的标题
                你要重点关注标题形式，同级的标题具有相同的形式（比如大小括号、中英文数字的使用等）
                注意标题JSON结构中的节点只能是给定的先验标题，不要自己生成新的标题，也不要对原标题字符进行任何修改或拆分。
                你的返回只能是JSON形式，该JSON结构的根节点固定为ROOT，所有其他节点的键是标题名称，值是下层节点；如下层没有其他标题，则其值固定为空字典。
                一个例子如下所示：
            """
        prompt = prompt + example + instruct 
    
    elif task=='judge-linkage':
        other_texts = kwargs['paras']['other_texts']
        prompt = f"""
                你会收到两段文字，分别来自连续文档的前一页的末段和下一页的首段：
                第一段文字'''{texts}'''
                第二段文字'''{other_texts}'''
                你需要判断，第二段文字是否可以和第一段文字合并为一段，你需要考虑两段文字内容的语义逻辑和连贯性，并使用0-10分打分，打分越高，代表两段文字越适合连接在一起。
                你的返回只能是JSON形式，其中键为"score"，值为你的打分。
            """
        keys_ = ['answer']

    elif task=='judge-answer':
        other_texts = kwargs['paras']['other_texts']
        prompt = f"""
                你会收到一个问题：{query}
                你还会收到两个答案。第一个是标准答案：{texts}；第二个是待评估的答案：{other_texts}
                你需要判断待评估答案与标准答案的符合程度，给出0-10的整数打分，打分越高，代表待评估答案越符合标准答案。
                注意，待评估答案不需要和标准答案完全一致，但关键数据和表达的意思必须符合。
                你的返回只能是JSON形式，其中键为"score"，值为你的打分。
            """
        keys_ = ['answer']
        
    elif task=='repair json':         
        prompt = f"""
                你会接收一段json字典形式的文本：{texts}.
                请你判断这个json的语法格式是否正确，如有错误，请进行修改并返回修改后的json字典。
                如无错误，直接返回原始json字典。
                无论如何，你的回答只能是json形式，不要返回其他多余信息。
            """
    else:
        pass
    
    response = call_with_messages(prompt, model, api_key)
    if response and response.status_code == HTTPStatus.OK:
        output = response.output
        if output and output.get("choices"):
            message = output["choices"][0].get("message")
            if message:
                answer = eval_response(message.get("content"), config=kwargs['config'])

    return answer



if __name__ == "__main__":
    
    
    print()
    
    


