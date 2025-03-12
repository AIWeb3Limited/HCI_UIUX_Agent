import http.client
import json
import re
import time

import json
import os
import re
import ast
import sys
import traceback
from io import StringIO
def message_template(role,new_info):
    new_dict={'role':role,'content':new_info}
    return new_dict
def chat_single(messages,mode="",model=None,verbose=False,temperature=0):
    conn = http.client.HTTPSConnection("api.openai-hub.com")
    headers = {
        'Authorization': 'Bearer sk-Nf0kLEmbRPRSFdD8qwlg1e7EHuoJMyaf1Z60Fh0IDLYosBEs',
        'Content-Type': 'application/json'
    }

    if mode=='json':
        payload = json.dumps({
            "model": "gpt-4o",
            "messages": messages,
            'temperature':temperature,
            "response_format": {"type": "json_object"}
        })

    elif mode == 'stream':
        payload = json.dumps({
            "model": "gpt-4o",
            "messages": messages,

            "stream": True
        })
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()

        if res.status != 200:
            print(f"Error: Received status code {res.status}")
            # print(res.read().decode("utf-8"))
            return None

        def response_generator():
            buffer = ""
            while True:
                chunk = res.read(1).decode("utf-8")  # Read one character at a time
                if not chunk:  # Break if the stream ends
                    break

                buffer += chunk
                if "\n" in buffer:  # Process lines one at a time
                    for line in buffer.split("\n"):
                        if line.strip():
                            try:
                                yield json.loads(line[5:])
                            except json.JSONDecodeError:
                                pass  # Skip lines that are not JSON
                    buffer = ""

        return response_generator()
    else:
        payload = json.dumps({
            "model": "gpt-4o",
            "messages": messages,
            'temperature': temperature,

        })
    # conn.request("POST", "/v1/chat/completions", payload, headers)
    # res = conn.getresponse()
    # data = res.read()
    # result = json.loads(data.decode("utf-8"))

    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY = 2  # 重试间隔（秒）

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            result = json.loads(data.decode("utf-8"))
            # print(result)
            final_result=result["choices"][0]["message"]["content"]
            # print(result)
            if mode=='json_few_shot':
                if verbose: print('json_few_shot',final_result)
                final_result=extract_words(final_result)
            return final_result

            # break  # 成功后退出循环
        except TimeoutError:
            print(f"请求超时，正在重试...（第 {attempt} 次尝试）")
            if attempt == MAX_RETRIES:
                print("达到最大重试次数，操作失败。")
                # 根据需要处理失败情况，比如抛出异常或记录日志
                raise
            time.sleep(RETRY_DELAY)  # 等待一段时间后重试
        finally:
            conn.close()  # 确保连接被关闭
def general_gpt_without_memory(query, messages=None,json_mode='',ask_prompt='',temperature=0,verbose=False):
    if isinstance(query, dict):
        query = str(query)
    if query == None:
        return None
    if messages == None:
        messages = []


    if messages == None:
        messages = []

    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    # result = chat_single(messages, '','gpt-4o-2024-05-13')
    result = chat_single(messages, json_mode,temperature=temperature,verbose=verbose)
    print('general_gpt result:', result)
    return result
def stream_api_test(query, messages=None,json_mode='',ask_prompt=''):

    if query == None:
        return None
    if messages == None:
        messages = []


    if messages == None:
        messages = []
    chunk_num = 0
    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    # result = chat_single(messages, '','gpt-4o-2024-05-13')
    chat_response =( chat_single(messages, 'stream'))
    chunk_num = 0
    in_code_block = False
    code_block_start = "```python"
    code_block_end = "```"
    buffer = ""
    line_buffer = ""
    total_buffer = ""
    total_char_list = []
    yield_list=[]
    chunk_num = 0
    for chunk in chat_response:
        if chunk is not None:
            if chunk["choices"][0]["delta"].get("content") is not None:
                if chunk_num == 0:
                    char = "\n" + chunk["choices"][0]["delta"]["content"]
                else:
                    char = chunk["choices"][0]["delta"]["content"]
                print(char, end="", flush=True)
                chunk_num += 1

    # for chunk in chat_response:
    #     if chunk is not None:
    #
    #         if json_part.choices[0].delta.content is not None:
    #             if chunk_num == 0:
    #                 char = "\n" + chunk.choices[0].delta.content
    #
    #             else:
    #                 char = chunk.choices[0].delta.content
    #             print(char)
# print(stream_api_test('who are you'))
def extract_code_blocks(code_str):
    code_blocks = []
    code_result = []
    if '```python' in code_str:
        parts = code_str.split("```python")
        for part in parts[1:]:  # 跳过第一个部分，因为它在第一个代码块之前
            code_block = part.split("```")[0]



            code_blocks.append(code_block)
        code_str=code_blocks[0]
        for code_part in code_str.split('\n'):
            if 'import' not in code_part and '=' not in code_part and code_part.strip() and 'print' not in code_part and '#' not in code_part:
                code_piece=f'print({code_part})'
            else:
                code_piece=code_part
            code_result.append(code_piece)
        # print(code_result)
        return "\n".join(code_result)
    return code_str
def messgae_initial_template(ask_prompt,query):
    messages=[]
    messages.append(message_template('system',ask_prompt))
    messages.append(message_template('user',query))
    return messages
def extract_words(text,mode='json'):
    # 使用正则表达式提取 JSON 部分
    if mode=='python':
        return extract_code_blocks(text)
    json_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)

    if not json_match:
        raise ValueError("No JSON data found in the text.")

    # 提取 JSON 字符串
    json_str = json_match.group(1)

    # 解析 JSON 字符串为 Python 字典
    data = json.loads(json_str)

    # 提取 'similar_words' 列表

    return data
def extract_python_code(input_str):
    # 使用正则表达式匹配 ```python ``` 包裹的代码块
    pattern = r"```python\s*\n(.*?)\n```"
    match = re.search(pattern, input_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No ```python``` code block found in the input string.")
def execute_and_display(code_str, local_vars=None):
    if local_vars is None:
        local_vars = {}

    # 将输出重定向到字符串缓冲区
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # 解析代码为 AST
        tree = ast.parse(code_str)

        # 如果 AST 的 body 为空，直接返回
        if not tree.body:
            output = sys.stdout.getvalue()
            return output if output else None

        # 分离最后一行（可能是表达式）和前面的语句
        if len(tree.body) > 1:
            exec(compile(ast.Module(tree.body[:-1], []), "<ast>", "exec"), local_vars)
            last_node = tree.body[-1]
        else:
            last_node = tree.body[0]

        # 如果最后一行是表达式，单独执行并获取结果
        if isinstance(last_node, ast.Expr):
            result = eval(compile(ast.Expression(last_node.value), "<ast>", "eval"), local_vars)
            # 获取缓冲区的输出（包括 print 调用和自动显示的内容）
            output = sys.stdout.getvalue()
            # 如果结果不是 None，附加到输出
            if result is not None and hasattr(result, "__str__"):
                output += str(result) + "\n"
            return output if output else result
        else:
            # 如果不是表达式，直接执行整个代码
            exec(code_str, local_vars)
            output = sys.stdout.getvalue()
            return output if output else None

    except Exception:
        # 捕获异常并返回完整的 Traceback 和之前的输出
        output = sys.stdout.getvalue()
        error_msg = traceback.format_exc()
        return output + error_msg if output else error_msg

    finally:
        # 恢复标准输出
        sys.stdout = old_stdout
def is_valid_variable_line(code_part):
    # 去除首尾空白字符
    code_part = code_part.strip()
    if not code_part:  # 如果是空行，返回 False
        return False

    # 分割成变量列表（按逗号分隔）
    parts = code_part.split(',')

    # 检查每个部分是否是合法变量名
    for part in parts:
        part = part.strip()  # 去除每个部分首尾的空白字符
        if not part:  # 如果部分为空（比如连续逗号或结尾逗号），返回 False
            return False

        # 检查变量名是否符合规则：字母或下划线开头，后面可以是字母、数字或下划线
        if not part[0].isalpha() and part[0] != '_':  # 首字符必须是字母或下划线
            return False

        # 检查剩余字符是否只包含字母、数字或下划线
        for char in part[1:]:
            if not (char.isalnum() or char == '_'):  # 只允许字母、数字、下划线
                return False

    return True