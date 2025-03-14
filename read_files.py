from fake_api import *
import sys
import time
import traceback
import io
import pandas
# import pandas as pd
import json
import pandas as pd

pd.set_option('display.max_rows', 2)  # 设置最大显示 10 行
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
    ask_prompt = """You are a data-processing intelligent assistant designed to answer questions about CSV files in the current directory. Your task is to analyze these files step-by-step using Python and pandas to provide data suitable for visualization (e.g., charts, maps), without generating the visualizations yourself. For each user query:

1. Understand the question and process the CSV data incrementally through multiple rounds of analysis.
2. In each round, unless code is unnecessary, write a Python snippet starting with ```python and ending with ```, using pandas to advance the solution.
3. Assume CSV files exist and can be read with `pd.read_csv("filename.csv")`.
4. After receiving execution results, decide whether further processing is needed or if the final answer is ready:
   - If more processing is required, explain the next steps clearly and provide the corresponding pandas code.
   - If the final answer is obtained, output the result with `final_answer` as the variable and print it out.
5. Structure the final output as concise JSON, containing only essential data for visualization, avoiding redundant fields.
6. Ensure code snippets are clear and well-commented to reflect your thought process.

Proceed with processing based on the user's query, following these steps."""
    # test_files = ["weather.csv", "events.csv"]
    file_short = csv_to_json(test_files)
    print(file_short)
    # return
    query_total = f"""
User files are located in ./uploads/, file name: {test_files}, their first two lines: {file_short}
User question: {query}
    """
    print(query_total)
    code_result = "python"
    messages = messages_initial_template(ask_prompt, query_total)
    local_vars = {}
    code_return = ''
    data_got_status=True
    round_num=0

    while True:
        round_num+=1
        code_result = chat_single(messages)
        print("response", code_result)

        # code_result = str_test

        messages.append(message_template('assistant', code_result))
        if 'python' in code_result:
            code_return = str(execute_and_display(extract_python_code(code_result), local_vars))
        else:
            code_return = code_result
        raw_return = str(code_return)
        if len(code_return) > 2500:

            code_return = code_return[:2500] + "..."

        print("code_return", code_return)
        messages.append(message_template('user', code_return))
        if '```python' not in code_result or 'final_answer' in code_result:
            if '```python' not in code_result:
                data_got_status=False
            if 'traceback' not in code_return.lower():

                break
    return raw_return[:4500],data_got_status


def html_generate_agent_modified(data_input, query):
    ask_prompt = """

You are an HTML-generating intelligent assistant (visualization agent). Your task is to generate corresponding HTML code to visualize data based on JSON-formatted data provided by the data processing agent. Users will indirectly provide questions and data through the data processing agent, and you need to:

Understanding Input Data:
Receive JSON-formatted data from the data processing agent.
Analyze the fields in the JSON, determine the data type, and identify visualization requirements.
Visualization Rules:
If the JSON contains latitude and longitude fields (e.g., longitude, latitude or lon, lat), use a Leaflet map (Using jsDelivr CDN) to display locations.
If the JSON contains date fields (e.g., date, dates), use a calendar format (CSS Grid creates a 7-column layout that simulates a calendar structure) for display.
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

If the input data contains multiple types, such as dates and locations, they should be organically combined. For example, clicking on a specific date should display the corresponding location on the map.

In the output explanation, only explain the received data without explaining the rendering of the HTML interface.
    """
    query_total = f"""
    data input:{data_input}
    query: {query}
    """
    print(query_total)

    messages = messages_initial_template(ask_prompt, query_total)
    code_result = chat_single(messages)
    print(code_result)
    return code_result


# aa="""
# ```python
# import pandas as pd
#
# # Load the weather data
# weather_df = pd.read_csv('./uploads/weather.csv')
#
# # Inspect unique weather symbols and basic statistics for temperature and wind speed
# unique_weather = weather_df['weather'].unique()
# temperature_stats = weather_df['average_temperature'].describe()
# wind_speed_stats = weather_df['average_wind_speed'].describe()
#
# unique_weather, temperature_stats, wind_speed_stats
# ```
# # """
# test_files = ["uploads/pollutionData204273.csv", "uploads/trafficData158324.csv"]
# query = "what's the relationship between pollution and traffic"
# aa,data_got_status= iterative_agent(query, test_files)
# html_generate_agent_modified(aa,query)
# print(execute_code(extract_words(aa,'python')))
# with open('./uploads/aarhus_parking (1).csv','r') as f:
#    for i in f:
#        print(i)
# Select relevant columns for the final JSON output
