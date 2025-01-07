from prompt_py import *
# from fake_api import *
from chat_py import *
from process_data import *
import re
from math import radians, cos, sin, sqrt, atan2
from geopy.distance import geodesic
def update_js_arrays(html_content, replacements):
    """
    批量更新 HTML/JavaScript 中数组的内容，并保持 JSON 格式的元素为合法的 JSON。

    参数:
    - html_content: str，HTML 内容。
    - replacements: str，JSON 格式的变量名和新值列表，例如：'[{"variable": "final_result", "value": ["a", "b", "c"]}]'

    返回:
    - 更新后的 HTML 内容。
    """
    # 将 JSON 格式字符串解析为 Python 列表
    try:
        replacement_list = json.loads(replacements)
    except json.JSONDecodeError:
        raise ValueError("输入的 replacements 必须是有效的 JSON 格式")

    # 遍历替换列表，逐一替换数组内容
    for item in replacement_list:
        variable_name = item.get("variable")
        new_value = item.get("value")
        if not variable_name or not isinstance(new_value, list):
            continue  # 跳过不完整的项目或非列表的值

        # 构造新的数组内容
        def format_element(element):
            if isinstance(element, (dict, list)):
                return json.dumps(element, ensure_ascii=False)  # 保持 JSON 格式
            else:
                return f'"{element}"'  # 字符串元素加引号

        new_array_content = ",\n".join([f'        {format_element(v)}' for v in new_value])
        replacement = f"const {variable_name} = [\n{new_array_content}\n        ];"

        # 正则表达式匹配原始数组
        pattern = rf'const\s+{variable_name}\s*=\s*\[.*?\];'
        html_content = re.sub(pattern, lambda m: replacement, html_content, flags=re.DOTALL)

    return html_content

def run_code_from_string(code_str):
    """
    去掉包含的代码块标记（```python 和 ```）并运行代码

    :param code_str: str，包含代码的字符串
    :return: None
    """
    # 去掉 ```python 和 ``` 标记
    if code_str.startswith("```python"):
        code_str = code_str[len("```python"):]
    # 直接移除最后一行的 ``` 标记
    lines = code_str.splitlines()
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    code_str = "\n".join(lines)

    # 去除可能的首尾空格
    code_str = code_str.strip()

    # 执行代码
    local_namespace = {}

    # 执行代码
    exec(code_str, globals(), local_namespace)
    return local_namespace.get("searched_result", None)

def know_data_agent(query,code_sample=None):
    messages=[]
    messages.append(message_template('system',know_data_prompt))
    messages.append(message_template('user',query))
    if code_sample:
        api_response=code_sample
    else:
        api_response=api_answer(messages,'json')

    know_data_agent_response=json.loads(api_response)
    # {
    #     "key_database":["events"],
    #     "related_databases": ["pollution", "weather"]
    # }
    all_data_bases=know_data_agent_response['key_database']+know_data_agent_response['related_databases']
    databases_info=[]
    for database_name in all_data_bases:
        databases_info.append(know_data(None,database_name))
    other_result={}
    for database_name in know_data_agent_response['related_databases']:
        other_result[database_name+"_data"]=get_data(None,database_name)

    return databases_info,other_result
def get_data_agent(databases_info,query,code_sample=None):
    messages=[]

    messages.append(message_template('system',get_data_prompt(databases_info)))
    messages.append(message_template('user',query))
    if code_sample:
        get_data_agent_response = code_sample
    else:
        get_data_agent_response=api_answer(messages)
    searched_result=run_code_from_string(get_data_agent_response)
    # searched_result
    return searched_result

def html_generate_agent(query,searched_result,other_result):
    messages=[]
    other_result_short=""
    if len(other_result)!=0:
        for databasename in other_result:
            other_result_short+=f"{databasename}=[{other_result[databasename][0]},...]\n"
            # other_result_short[databasename]=[other_result[databasename][0]]
    variables_list=["searched_result"]
    variables_list.extend(list(other_result.keys()))
    html_prompt=get_html_generate_prompt(query,str(searched_result[0])+",...",other_result_short,variables_list)
    print("html_prompt", html_prompt)
    messages.append(message_template('system',html_prompt))
    messages.append(message_template('user',query))
    html_generate_agent_response=api_answer(messages)
    replacements_list=[]
    replacements_list.append({'variable':"searched_result","value":searched_result})
    for var_name in other_result:
        replacements_list.append({'variable':var_name, "value": other_result[var_name]})
    print("replacements_list",replacements_list)
    replacements_json =json.dumps(replacements_list)
    new_html=update_js_arrays(html_generate_agent_response,replacements_json)
    return new_html,html_generate_agent_response,replacements_json

def generate_response(query):
    code_sample = """```python
    
good_weather = get_data(['☀️', '🌤️'], 'weather')

# 提取日期列表
good_weather_dates = [entry['date'] for entry in good_weather]

# 查询污染小的数据，假设污染小的条件是平均颗粒物小于80
low_pollution = get_data(['Good'], 'pollution')

# 提取日期列表
low_pollution_dates = [entry['date'] for entry in low_pollution]

# 找到天气好且污染小的日期
good_weather_low_pollution_dates = set(good_weather_dates) & set(low_pollution_dates)

# 查询活动数据
searched_result = []
for date in good_weather_low_pollution_dates:
    activities = get_data(date, 'events')
    searched_result.extend(activities)

```
    """
    data_sample = """
{
    "key_database": ["events"],
    "related_databases": ["pollution", "weather"]
}
    """
    # query = '我想知道污染程度随天气的变化关系'
    databases_info, other_result = know_data_agent(query)
    searched_result = get_data_agent(databases_info, query)
    # other_result_short = {}
    # for databasename in other_result:
    #     other_result_short[databasename] = [other_result[databasename][0]]
    # print(other_result_short)
    print("searched_result", searched_result)
    print("other_result", other_result)
    new_html,html_generate_agent_response,replacements_json = html_generate_agent(query, searched_result, other_result)
    print(new_html)
    return new_html,html_generate_agent_response,replacements_json
# generate_response('我想知道天气好污染小的时候有啥活动')
# good_weather = get_data(['☀️', '🌤️'], 'weather')
#
# # Step 2: Get good pollution data
# good_pollution = get_data(['Good'], 'pollution')
#
# # Step 3: Extract dates with good weather
# good_weather_dates = [entry['date'] for entry in good_weather]
#
# # Step 4: Extract dates with good pollution
# good_pollution_dates = [entry['date'] for entry in good_pollution]
#
# # Step 5: Find common dates with both good weather and good pollution
# common_good_dates = set(good_weather_dates).intersection(good_pollution_dates)
# print(common_good_dates)
# # Step 6: Get events on common good dates
# searched_result = get_data(list(common_good_dates), 'events')
# event_title = 'Tegnearrangement med Mark Bundgaard'
# event_data = get_data([event_title], 'events')
#
# # Step 2: Extract the latitude and longitude of the event
# event_latitude = event_data[0]['latitude']
# event_longitude = event_data[0]['longitude']
#
# # Step 3: Get all parking data
# parking_data = get_data([], 'parking')

# Step 4: Calculate the distance from the event to each parking location


# def calculate_distance(event_location, parking_location):
#     return geodesic(event_location, parking_location).kilometers
#
# event_location = (event_latitude, event_longitude)
# parking_distances = []
#
# for parking in parking_data:
#     parking_location = (parking['latitude'], parking['longitude'])
#     distance = calculate_distance(event_location, parking_location)
#     parking_distances.append((distance, parking))
#
# # Step 5: Sort the parking locations by distance
# parking_distances.sort(key=lambda x: x[0])
#
# # Step 6: Select the 5 closest parking locations
# closest_parkings = parking_distances[:5]
#
# # Step 7: Extract the parking data for the 5 closest locations
# searched_result = [parking[1] for parking in closest_parkings]
# print(searched_result)
# Step 1: Search for the event with the title 'Tegnearrangement med Mark Bundgaard'
# event_title = 'Tegnearrangement med Mark Bundgaard'
# event_data = get_data([event_title], 'events')
#
# # Step 2: Extract the longitude and latitude of the event
# event_longitude = event_data[0]['longitude']
# event_latitude = event_data[0]['latitude']
#
# # Step 3: Get all parking data
# parking_data = get_data(None, 'parking')
# print(parking_data)
# # Step 4: Calculate the distance from the event to each parking location
# from geopy.distance import geodesic
#
# def calculate_distance(event_location, parking_location):
#     return geodesic(event_location, parking_location).kilometers
#
# event_location = (event_latitude, event_longitude)
# parking_distances = []
#
# for parking in parking_data:
#     parking_location = (parking['latitude'], parking['longitude'])
#     distance = calculate_distance(event_location, parking_location)
#     parking_distances.append((distance, parking))
#
# # Step 5: Sort the parking locations by distance and select the 5 closest
# parking_distances.sort(key=lambda x: x[0])
# closest_parkings = parking_distances[:5]
#
# # Step 6: Extract the parking data for the 5 closest locations
# searched_result = [parking[1] for parking in closest_parkings]
# event_title = 'Tegnearrangement med Mark Bundgaard'
# event_data = get_data([event_title], 'events')
#
# # Step 2: Extract the longitude and latitude of the event
# event_longitude = event_data[0]['longitude']
# event_latitude = event_data[0]['latitude']
#
# # Step 3: Get all parking data
# parking_data = get_data([], 'parking')
# print(parking_data)
# # Step 4: Calculate the distance from the event to each parking location
# from geopy.distance import geodesic
#
# def calculate_distance(event_location, parking_location):
#     return geodesic(event_location, parking_location).kilometers
#
# event_location = (event_latitude, event_longitude)
# parking_distances = []
# print(event_location)
# for parking in parking_data:
#     parking_location = (parking['latitude'], parking['longitude'])
#     distance = calculate_distance(event_location, parking_location)
#     parking_distances.append((distance, parking))
#
# # Step 5: Sort the parking locations by distance and select the 5 closest
# parking_distances.sort(key=lambda x: x[0])
# closest_parkings = parking_distances[:5]
#
# # Step 6: Extract the parking data for the 5 closest locations
# searched_result = [parking[1] for parking in closest_parkings]
# print(closest_parkings)
# event_title = "Tegnearrangement med Mark Bundgaard"
# event_data = get_data(['Tegnearrangement med Mark Bundgaard'], 'events')
#
# print(event_data)
# print(know_data(None,'events'))
# all_weather_data = get_data([ 'date','average_temperature'], 'weather')
#
# # Step 2: Get all pollution data
# all_pollution_data = get_data(['date', 'average_particullate_matter'], 'pollution')
#
# # Step 3: Merge data based on date
# merged_data = []
# for weather in all_weather_data:
#     for pollution in all_pollution_data:
#         if weather['date'] == pollution['date']:
#             merged_entry = {
#                 'date': weather['date'],
#                 'average_temperature': weather['average_temperature'],
#                 'average_particullate_matter': pollution['average_particullate_matter']
#             }
#             merged_data.append(merged_entry)
# print(all_weather_data)