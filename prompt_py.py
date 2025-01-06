
def get_html_generate_prompt(query,searched_result,other_results,variables):
    html_generate_prompt=f"""
    
Create an HTML page to display data dynamically based on the following variables:

The user wants to search for {query}.
The system retrieves a list of JSON to be highlighted:
searched_result = [{searched_result}]
other_results,displayed in a dark-themed style:
{other_results}

The page must include:

Smooth animations and interactive clickable elements.
Hover effects, rounded corners, soft shadows, grid lines, and emojis for a modern UI.
A calendar-style layout to represent each day.
Tooltips or pop-up bubbles to display detailed information (e.g., map data)(Ensure the popup bubble is on the topmost layer.).
Highlighted styling for searched_result data, with other_results styled in a darker theme.
Leaflet integration for map-related features.
Ensure the layout is contained within a manageable-sized container, suitable for embedding in larger layouts.

adhering strictly to the provided variable names: {variables}. The HTML page must dynamically adjust its content based on the variable values, meaning replacing these values will automatically update the display.

Provide only the HTML code, formatted professionally, with the provided variable.
    """
    return html_generate_prompt
def get_data_prompt(data_info):
    data_prompt=f"""
    我有下面几个数据库，他们每个键的数据类型和范围如下所示：
    {data_info}
    如果你想查询数据，你可以通过调用函数get_data(查询数据名称，数据库名称)
    数据名称可以是日期，或者具体的数据名称，输入格式可以是列表
    return的是查询到的对应的数据列表，列表的每个元素是有相同键名的字典
    比如good_weather=get_data(['☀️','🌤️'],'weather')
    return：got corresponding data：overview:{{'length': 14, 'keys': dict_keys(['date', 'weather', 'average_temperature', 'average_wind_speed'])}}
    你负责查到对应的数据，如果这个查询不能同时完成，比如后面的查询需要依赖前面的查询结果，那么需要按照依赖关系写出合理的查询程序
    只回答代码，每行函数都要有变量
    最终得出的数据变量是searched_result,是搜索到的数据列表，和get_data的返回格式一致
    """
    return data_prompt

know_data_prompt="""
    我有下面几个数据库
    events：city,title,price,library,longitude,latitude,date
    pollution：longitude,latitude,date,average_particullate_matter,pollution_condition
    weather：date,weather,average_temperature,average_wind_speed
    parking：date,longitude,latitude,garagecode,occupancy rate
    
    你的职责是根据用户需求返回一个关键和相关的数据库列表，以json格式返回
    用户：我想知道天气好和污染小的时候有什么活动
    Return: 
    {
        "key_database":["events"],
        "related_databases": ["pollution", "weather"]
    }
    用户：我想知道天气温度和污染物浓度的关系
    Return: 
    {
        "key_database":["pollution", "weather"],
        "related_databases": []
    }
    用户：我想知道距离某个活动最近的5个停车场
    Return: 
    {
        "key_database":["events","parking"],
        "related_databases": []
    }
    """

