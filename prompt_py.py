
def get_html_generate_prompt(query,searched_result,other_results,variables,route_plan):


    if other_results!="":
        other_results_str=f"""

    {other_results}
        Highlighted styling for searched_result data,other results also need to be displayed
        """
    else:
        other_results_str=''
    if route_plan:
        result_show_method=f"""
    Design a user interface panel for visualizing the path planning results, which are structured as follows:
    searched_result = [{searched_result}].
    
    The panel should include the following functionalities and features:
    
    Route Ranking Display:
    
    The path planning results are composed of routes grouped by THREE different ranks, other two ranks also need to display.
    Include buttons or toggles to switch between the display of different rank routes. 注意确保在切换路线时不会重复初始化地图容器。
    Route Overview Details:
    
    For each rank, display the following aggregate metrics:
    Average Velocity
    Total Distance
    Total Duration
    Interactive Route Elements:
    
    Clicking on any individual route segment should display:
    Velocity
    REPORT_ID
    DURATION_IN_SEC
    Path Visualization:
    
    Draw each route segment by connecting its POINT_1 and POINT_2 using width 18 pixels straight lines (based on their latitude and longitude).
    Assign transparent value to each segment based on its VELOCITY (color red):
    Higher velocity = Lower transparent.
    Lower velocity = Higher transparent.
    
    Ensure the panel design is user-friendly, clearly labeled, and effectively communicates the information.
    And also show Start node and End node in map.
        """
    else:
        result_show_method=f"""
    The system retrieves a list of JSON to be highlighted:
    searched_result = [{searched_result}]

    {other_results_str}
        """

    html_generate_prompt=f"""
        
    Create an HTML page to display data dynamically based on the following variables:
    
    The user wants to search for {query}.

    {result_show_method}
    The page must include:
    
    Smooth animations and interactive clickable elements.
    Hover effects, rounded corners, soft shadows, grid lines, and emojis for a modern UI.

    Tooltips or pop-up bubbles to display detailed information (e.g., map data)(Ensure the popup bubble is on the topmost layer.).

    Ensure the layout is contained within a manageable-sized container, suitable for embedding in larger layouts.
    
    adhering strictly to the provided variable names: {variables}. The HTML page must dynamically adjust its content based on the variable values, meaning replacing these values will automatically update the display.
    You need to configure the page layout based on the user's requirements. For example, if the user requests a line chart, you should provide a line chart.
    Provide only the HTML code, formatted professionally, with the provided variable.
        """

    return html_generate_prompt
def get_data_prompt(data_info,route_plan):
    if route_plan:
        route_plan_prompt="""
        用户需要路径规划时，可以调用函数
        searched_result=plan_routes(start_longitude, start_latitude, end_longitude, end_latitude)
    
        """
    else:
        route_plan_prompt="""
    如果要查找某个地方附近的停车场，可以直接调用
    get_parking(place_name, (latitude, longitude),number)
    Return [parking list..]
        """
    data_prompt=f"""
    我有下面几个数据库，他们每个键的数据类型和范围如下所示：
    {data_info}
    如果你想查询数据，你可以通过调用函数get_data(查询数据名称，数据库名称)
    数据名称可以是日期，或者具体的数据名称，输入格式可以是列表
    return的是查询到的对应的数据列表，列表的每个元素是有相同键名的字典
    比如good_weather=get_data(['☀️','🌤️'],'weather')
    Return：got corresponding data：overview:{{'length': 14, 'keys': dict_keys(['date', 'weather', 'average_temperature', 'average_wind_speed'])}}
    比如查找名为aaa的活动，event_data = get_data(['aaa'], 'events')
    Return [{{'city': 'bbb', 'title': 'aaa', 'price': '0', 'library': 'Tilst Bibliotek', 'longitude': 10.111042, 'latitude': 56.18945, 'date': '2013-10-28'}}]
    

    
    {route_plan_prompt}
    
    你负责查到对应的数据，如果这个查询不能同时完成，比如后面的查询需要依赖前面的查询结果，那么需要按照依赖关系写出合理的查询程序。如果用户需要绘制图，你不要写出绘制代码，只是提供绘制的数据
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
    route_plan
    你的职责是根据用户需求返回一个关键和相关的数据库列表，以json格式返回
    
    用户：我想知道天气好和污染小的时候有什么活动
    Return: 
    {
        "key_database":["events"],
        "related_databases": ["pollution", "weather"],
        "route_plan":false
    }
    用户：我想知道天气温度和污染物浓度的关系
    Return: 
    {
        "key_database":["pollution", "weather"],
        "related_databases": [],
        "route_plan":false
    }
    用户：我想知道距离某个活动最近的5个停车场
    Return: 
    {
        "key_database":["events","parking"],
        "related_databases": [],
        "route_plan":false
    }
    用户：帮我规划某个活动到某个停车场的路线
    Return: 
    {
        "key_database":["events","parking"],
        "related_databases": [],
        "route_plan":true
    }
    """

