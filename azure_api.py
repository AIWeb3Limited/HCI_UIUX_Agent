import json
import os
import re
import ast
import sys
from io import StringIO
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()
client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
  api_version="2024-02-01"
)



def message_template(role,new_info):
    new_dict={'role':role,'content':new_info}
    return new_dict


def chat_single(messages, mode="", model='gpt-4o', temperature=0,verbose=False):
    if mode == "json":

        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
            messages=messages
        )
    elif mode == 'stream':
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            stream=True,
            max_tokens=2560

        )
        return response
    elif mode == 'json_few_shot':
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=2560

        )
        result= response.choices[0].message.content

        if verbose:print(result)

        return extract_json_and_similar_words(
            result)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,

        )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def general_gpt_without_memory(query, messages=None, json_mode='', ask_prompt='',verbose=False):
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
    result = chat_single(messages, json_mode,verbose=verbose)
    print('general_gpt result:', result)
    return result

def format_list_string(input_str):
    # 正则匹配大括号内的内容
    match = re.search(r'\{\s*"[^"]+"\s*:\s*\[(.*?)\]\s*\}', input_str)
    if not match:
        return "Invalid input format"

    list_content = match.group(1)  # 获取匹配到的列表内容
    elements = [e.strip() for e in list_content.split(',')]  # 拆分并去除多余空格

    formatted_elements = []
    for elem in elements:
        if not re.match(r'^([\'"])(.*)\1$', elem):  # 检查是否被引号包裹
            elem = f'"{elem}"'  # 添加双引号
        formatted_elements.append(elem)

    return f'{{ "similar_words":[{", ".join(formatted_elements)}]}}'
def extract_json_and_similar_words(text):
    try:

        # 使用正则表达式提取 JSON 部分
            json_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)

            if not json_match:
                raise ValueError("No JSON data found in the text.")

            # 提取 JSON 字符串
            json_str = json_match.group(1)
            print(json_str)
            # 解析 JSON 字符串为 Python 字典
            data = json.loads(json_str)

            # 提取 'similar_words' 列表

            return data
    except Exception as e:
        print( e)
        if 'similar_words' in text:
            json_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)

            if not json_match:
                raise ValueError("No JSON data found in the text.")

            # 提取 JSON 字符串
            json_str = json_match.group(1)

            # 解析 JSON 字符串为 Python 字典
            data = json.loads(format_list_string(json_str))

        # 提取 'similar_words' 列表

        return data


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
            # 匹配只包含变量名的行（只包含字母、数字、下划线，且不含运算符等）
            if is_valid_variable_line(code_part):
                code_piece = f'print({code_part.strip()})'
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
            return None

        # 分离最后一行（可能是表达式）和前面的语句
        if len(tree.body) > 1:
            exec(compile(ast.Module(tree.body[:-1], []), "<ast>", "exec"), local_vars)
            last_node = tree.body[-1]
        else:
            last_node = tree.body[0]

        # 如果最后一行是表达式（如 pollution_data.head()），单独执行并获取结果
        if isinstance(last_node, ast.Expr):
            result = eval(compile(ast.Expression(last_node.value), "<ast>", "eval"), local_vars)
            # 如果有输出（比如 print 调用），获取缓冲区的输出
            output = sys.stdout.getvalue()
            if output:
                print(output, end="")
            # 如果结果不是 None，显示它
            if result is not None:
                # 对于 Pandas DataFrame/Series，调用其 __str__ 方法
                if hasattr(result, "__str__"):
                    print(result)
            return result
        else:
            # 如果不是表达式，直接执行整个代码
            exec(code_str, local_vars)
            output = sys.stdout.getvalue()
            if output:
                print(output, end="")
            return None

    finally:
        # 恢复标准输出
        sys.stdout = old_stdout
