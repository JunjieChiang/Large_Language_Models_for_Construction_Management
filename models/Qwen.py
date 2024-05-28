import dashscope
from .Model import Model
from http import HTTPStatus


class Qwen(Model):
    def get_completion(self, prompt):
        model = self.name
        api_keys = self.api_keys
        # temperature = self.temperature
        # gpu = self.gpus

        messages = [
            {'role': 'system', 'content': '你是ChatRevit，一个BIM模型属性信息查询助手'},
            {'role': 'user', 'content': prompt},
        ]

        response = dashscope.Generation.call(
            model=model,
            api_key=api_keys,
            messages=messages,
            result_format='message'  # set the result to be "message" format.
        )

        if response.status_code == HTTPStatus.OK:
            content = response['output']['choices'][0]['message']['content']

            return content

        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
