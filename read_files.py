from azure_api import *
import sys
import time
import traceback
import io
import pandas
import pandas as pd
import json

output = io.StringIO()
original_stdout = sys.stdout


def csv_to_json(file_list):
    result = {}
    for file_name in file_list:
        # 读取 CSV 文件的前两行
        df = pd.read_csv(file_name, nrows=2)

        # 将 DataFrame 转换为列表形式（每行是一个字典）
        result[file_name] = df.to_dict(orient='records')
    # 转换为 JSON 字符串
    return json.dumps(result, indent=2)


# 测试代码

def execute_code(code_str):
    sys.stdout = output

    start_time = time.time()  # 记录函数开始时间

    try:
        exec(code_str, globals())
    except Exception as e:
        exc_info = traceback.format_exc()

        print(f"An error occurred:\n{exc_info}")

    end_time = time.time()  # 记录函数结束时间
    run_time = end_time - start_time

    code_result = output.getvalue().replace('\00', '')
    output.truncate(0)
    sys.stdout = original_stdout
    print(code_result)
    return code_result


def iterative_agent(query, test_files, str_test=None):
    ask_prompt = """
You are a data-processing intelligent assistant. Users will ask you questions about the contents of CSV files in the current directory, requiring you to solve problems through multiple rounds of analysis and computation. Your tasks include:

Understanding the user's question about the CSV file and analyzing it step by step.
In each round, writing a Python snippet using pandas to process the CSV file and advance the solution.
Receiving the execution results of each pandas code snippet.
Deciding whether to write another pandas code snippet for further processing or if the final answer has been obtained.
If further processing is needed, clearly explaining the next steps and writing the corresponding pandas code.
If the final answer is obtained, outputting a well-structured JSON format result with final_answer: <JSON result> and stopping further processing. The JSON output should:
Be clearly structured, containing only the essential data needed to answer the question.
Be suitable for various visualization methods (e.g., charts, maps).
Avoid redundant fields, keeping only the necessary information.
Each code snippet should be clearly written to help users understand your thought process.
Assume that the mentioned CSV files exist in the current directory and can be read using pandas (e.g., pd.read_csv("filename.csv")).
Now, proceed with processing based on the user's query following these steps.
Please only give one piece of python code at one session.
    """
    # test_files = ["weather.csv", "events.csv"]
    file_short = csv_to_json(test_files)
    query_total = f"""
    用户文件都是当前路径，文件名：{test_files}，他们的前两行：{file_short}
    用户问题：{query}
    """
    print(query_total)
    code_result = ""
    messages = messgae_initial_template(ask_prompt, query_total)
    code_return = ''
    while 'final_answer' not in code_result:
        code_result = chat_single(messages)
        # code_result = str_test
        print("response", code_result)
        messages.append(message_template('assistant', code_result))
        code_return = execute_and_display(code_result)
        print("code_return", code_return)
        messages.append(message_template('user', code_return))
    return code_return


def html_generate_agent(data_input, query):
    ask_prompt = """
 
You are an HTML-generating intelligent assistant (visualization agent). Your task is to generate corresponding HTML code to visualize data based on JSON-formatted data provided by the data processing agent. Users will indirectly provide questions and data through the data processing agent, and you need to:

Understanding Input Data:
Receive JSON-formatted data from the data processing agent.
Analyze the fields in the JSON, determine the data type, and identify visualization requirements.
Visualization Rules:
If the JSON contains latitude and longitude fields (e.g., longitude, latitude or lon, lat), use a Leaflet map to display locations.
If the JSON contains date fields (e.g., date, dates), use a calendar format (based on simple HTML and CSS or a lightweight library like FullCalendar) for display.
For other data types (e.g., numerical or categorical data), use Chart.js to generate bar charts or line charts by default.
If the data type is unclear, choose a reasonable default visualization method (such as a table).
HTML Generation Requirements:
Embed the generated HTML code into a container with the following styles:
Shadow: Add a CSS shadow effect to the container (e.g., box-shadow).
Buttons: Include at least one interactive button (e.g., "Refresh" or "Switch View") with a hover effect.
Animations: Add simple CSS animations (e.g., fade-in effects) to the container or visualization elements.
Ensure the HTML code is complete, including the necessary external libraries (such as Leaflet, Chart.js, or FullCalendar via CDN) in the <head> section, along with inline CSS/JavaScript.
The code should be able to run directly in a browser.
Output Format:
Wrap the generated HTML code with triple backticks (```html as the first line and ``` as the last line).
Provide a brief explanation of the HTML's functionality and the visualization method used before the code.

    """
    query_total = f"""
    data input:{data_input}
    query: {query}
    """
    print(query_total)

    messages = messgae_initial_template(ask_prompt, query_total)
    code_result = chat_single(messages)
    return code_result


# aa="""
# code_return {'good_weather_events': [{'date': '2014-08-01', 'city': 'Aarhus', 'title': 'International Playgroup', 'library': 'Hovedbiblioteket', 'longitude': 10.200179, 'latitude': 56.156617}, {'date': '2014-08-09', 'city': 'Viby J', 'title': 'Bustur til Store Bogdag ved Hald Hovedgaard lÃ¸rdag 9. august', 'library': 'Viby Bibliotek', 'longitude': 10.164431, 'latitude': 56.130402}, {'date': '2014-08-09', 'city': 'Aarhus', 'title': 'ByggelÃ¸rdag', 'library': 'Hovedbiblioteket', 'longitude': 10.200179, 'latitude': 56.156617}, {'date': '2014-08-01', 'city': 'Viby J', 'title': 'Artmoney â\x80?penge med dobbeltvÃ¦rdi ', 'library': 'Viby Bibliotek', 'longitude': 10.164431, 'latitude': 56.130402}]}
#
# """
file_list = [r'uploads\pollutionData204273.csv', r'uploads\trafficData158324.csv']
query = "what's the pollution and traffic in different time periods?"
aa = iterative_agent(query, file_list)
print(aa)
# html_generate_agent(aa,query)
# print(execute_code(extract_words(aa,'python')))/
