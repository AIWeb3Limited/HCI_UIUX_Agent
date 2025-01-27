import json
from route_plan import *
import pandas as pd
import numpy as np
import csv
from geopy.distance import geodesic
def know_data( column_names,csv_file):
    """
    获取指定CSV文件中指定列的数据类型和数据范围。

    参数：
        csv_file (str): CSV文件的名称或路径。
        column_names (list): 需要分析的列名列表。

    返回：
        dict: 包含每个列的数据类型和数据范围的信息。
    """
    csv_file=f"available_data/{csv_file}.csv"
    data_range_string='data_range'

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        if not column_names:
            column_names = df.columns.tolist()

        if isinstance(column_names, str):
            column_names = [column_names]
        # 检查列名是否存在
        missing_columns = [col for col in column_names if col not in df.columns]
        if missing_columns:
            raise ValueError(f"以下列名不存在于CSV文件中: {missing_columns}")

        result = {csv_file.replace(".csv",'').split('/')[1]:{}}

        for column_name in column_names:
            # 获取列的数据类型
            column_data = df[column_name]
            data_type = column_data.dtypes

            # 检查是否为日期格式字符串
            if data_type == 'object':
                try:
                    parsed_dates = pd.to_datetime(column_data, format='%Y-%m-%d', errors='coerce')
                    if parsed_dates.notna().all():
                        mapped_type = 'string(date)'
                        data_range = (
                            parsed_dates.min().strftime('%Y-%m-%d'),
                            parsed_dates.max().strftime('%Y-%m-%d')
                        )
                    else:
                        raise ValueError
                except ValueError:
                    mapped_type = 'string'
                    unique_values = column_data.dropna().unique()
                    if len(unique_values) <= 10:
                        data_range = list(unique_values)
                        data_range_string = 'All data type'

                    else:
                        sample_values = column_data.dropna().unique()[:3]
                        data_range = list(sample_values) if len(sample_values) > 0 else (None, None)
                        data_range_string = 'sampled three data'
            elif np.issubdtype(data_type, np.number):
                data_range_string = 'data_range'

                if np.issubdtype(data_type, np.integer):
                    mapped_type = 'int'
                else:
                    mapped_type = 'float'
                data_range = (column_data.min(), column_data.max())

            elif np.issubdtype(data_type, np.datetime64):
                data_range_string = 'data_range'

                mapped_type = 'string(date)'
                data_range = (
                    column_data.min().strftime('%Y-%m-%d') if pd.notnull(column_data.min()) else None,
                    column_data.max().strftime('%Y-%m-%d') if pd.notnull(column_data.max()) else None
                )
            else:
                mapped_type = '未知'
                sample_values = column_data.dropna().unique()[:3]
                data_range = list(sample_values) if len(sample_values) > 0 else (None, None)
                data_range_string='sampled three data'

            result[csv_file.replace(".csv",'').split('/')[1]][column_name] = {
                'data_type': mapped_type,
                data_range_string: data_range
            }

        print(result)
        return result

    except FileNotFoundError:
        return {"error": f"文件 '{csv_file}' 未找到。"}
    except ValueError as e:
        return {"error": str(e)}



def get_data(data_names, csv_filename):
    """
    从CSV文件中查找数据名称并返回对应行或列的数据，返回格式为JSON。

    :param data_names: 单个数据名称或数据名称列表
    :param csv_filename: CSV文件名（不含扩展名）
    :return: JSON格式的结果
    """
    # 确保 data_names 是列表
    csv_filename = f"available_data/{csv_filename}.csv"

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_filename, encoding='utf-8-sig')

        if isinstance(data_names, str):
            data_names = [data_names]
        if not data_names or len(data_names)==0:
            raw_data = df.to_dict(orient='records')
            data_intro = {"length": len(raw_data), "keys": list(df.columns)}
            print(data_intro)
            return raw_data
        # 检查是否所有的 data_names 都是列名
        if all(name in df.columns for name in data_names):
            # 如果 data_names 是列名，返回指定的列
            selected_columns = df[data_names]

            # 检查是否有 'date' 列
            if 'date' in df.columns:
                selected_columns['date'] = df['date']
            else:
                print("Warning: 'date' column not found in the CSV file.")

            # 数据简介
            data_intro = {'length': len(selected_columns), 'keys': list(selected_columns.columns)}
            print(data_intro)

            # 返回包含 date 的结果
            return selected_columns.to_dict(orient='records')

        # 如果 data_names 为空，返回所有数据


        # 查找包含指定数据名称的行
        matched_rows = df[df.apply(lambda row: any(data_name in row.values for data_name in data_names), axis=1)]
        print(matched_rows)
        # 如果有匹配的行，返回数据和基本信息
        if not matched_rows.empty:
            raw_data = matched_rows.to_dict(orient='records')
            data_intro = {'length': len(raw_data), 'keys': list(matched_rows.columns)}
            print(data_intro)
            return raw_data
        else:
            data_intro = {'length': 0}
            print(data_intro)
            return []

    except FileNotFoundError:
        return json.dumps({"error": "CSV file not found."}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
def get_parking(name,address,n):
    latitude, longitude=address
    """
    Find the nearest n locations to the given latitude and longitude.

    Parameters:
        file_path (str): Path to the CSV file containing 'latitude' and 'longitude' columns.
        latitude (float): Latitude of the input location.
        longitude (float): Longitude of the input location.
        n (int): Number of nearest locations to return.

    Returns:
        list: A JSON-compatible list of dictionaries representing the nearest n rows.
    """
    # Load the CSV file
    df = pd.read_csv('available_data/parking.csv')

    # Ensure the required columns exist
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError("The CSV file must contain 'latitude' and 'longitude' columns.")

    # Calculate distances
    def calculate_distance(row):
        return geodesic((latitude, longitude), (row['latitude'], row['longitude'])).kilometers

    df['distance'] = df.apply(calculate_distance, axis=1)

    # Sort by distance and select the top n rows
    nearest_locations = df.nsmallest(n, 'distance')

    result = [{
        "reference_point_name": name,
        "reference_point_location": {"latitude": latitude, "longitude": longitude},
        "nearest_locations": nearest_locations.to_dict(orient='records')
    }]

    return result



def plan_routes(start_longitude, start_latitude, end_longitude, end_latitude):
    best_3_routes = plan_routes_function(start_longitude, start_latitude, end_longitude, end_latitude, k=3)
    return best_3_routes
# get_data("Viby Bibliotek",'events')
# start_longitude, start_latitude = 10.21284, 56.16184  # 北京天安门
# end_longitude, end_latitude = 10.164431,56.130402 # 北京某地示例
# #
# best_3_routes = plan_routes( start_longitude, start_latitude, end_longitude, end_latitude, k=3)
# # 使用示例
    # except Exception as e:
    #     return json.dumps({"error": str(e)}, ensure_ascii=False)

# # Step 1: 查询天气好的数据
# # 查询天气好的数据
# good_weather = get_data(['☀️', '🌤️'], 'weather')
#
# # 筛选天气好的日期
# good_weather_dates = [entry['date'] for entry in good_weather]
#
# # 查询污染不严重的数据
# low_pollution = get_data(['Good'], 'pollution')
#
# # 筛选污染不严重的日期
# low_pollution_dates = [entry['date'] for entry in low_pollution]
#
# # 找到天气好且污染不严重的公共日期
# good_dates = list(set(good_weather_dates) & set(low_pollution_dates))
#
# # 查询这些日期对应的活动
# final_result = get_data(good_dates, 'events')
# print(final_result)
# # pollution=get_data('2014-08-01','pollution')
# wea=get_data(None,'pollution')
# # print(pollution[0])
# print(wea)
#
# get_data('Good', 'weather')
# get_data('Good', 'weather')
# # # print(know_data(final_result[0].keys(),'events'))
# library_location = get_data(['Tilst Bibliotek'], 'events')
# # Assuming the location details are in the first element of the list
# tilst_bibliotek_location = library_location[0]
#
# # Step 2: Use the latitude and longitude from Tilst Bibliotek to find parking nearby
# latitude = tilst_bibliotek_location['latitude']
# longitude = tilst_bibliotek_location['longitude']
#
# searched_result = get_parking('name',(latitude, longitude), 5)
# print(searched_result)