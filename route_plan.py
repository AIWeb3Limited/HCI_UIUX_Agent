import csv
import math
import networkx as nx

csv_file = r"available_data/trafficMetaData.csv"


def round_json_values(data):
    for route in data:
        for key, value in route.items():
            if isinstance(value, list):
                # Recursively process the list
                for item in value:
                    for sub_key, sub_value in item.items():
                        # Check the conditions: not ending in LAT or LNG, not string or integer
                        if (not sub_key.endswith('_LAT') and not sub_key.endswith('_LNG') and
                            isinstance(sub_value, (float, int)) and not isinstance(sub_value, int)):
                            item[sub_key] = round(sub_value, 1)
            else:
                # Check the conditions for the top-level keys
                if (not key.endswith('_LAT') and not key.endswith('_LNG') and
                    isinstance(value, (float, int)) and not isinstance(value, int)):
                    route[key] = round(value, 1)
    return data

def haversine_distance(lng1, lat1, lng2, lat2):
    """
    根据经纬度计算两点之间的球面距离（米）。
    """
    R = 6371_000  # 地球平均半径，单位：米
    rad = math.pi / 180
    d_lat = (lat2 - lat1) * rad
    d_lng = (lng2 - lng1) * rad
    a = (math.sin(d_lat/2))**2 + math.cos(lat1*rad)*math.cos(lat2*rad)*(math.sin(d_lng/2))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

def preload_csv(csv_file_path):
    """
    将 CSV 文件中的路线数据一次性读入内存。
    返回包含路线信息的列表。
    """
    edges = []
    with open(csv_file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p1_lng = float(row['POINT_1_LNG'])
            p1_lat = float(row['POINT_1_LAT'])
            p2_lng = float(row['POINT_2_LNG'])
            p2_lat = float(row['POINT_2_LAT'])
            dist = float(row['DISTANCE_IN_METERS'])
            dur = float(row['DURATION_IN_SEC'])
            report_id = row['REPORT_ID']
            velocity = dist / dur if dur > 0 else 0
            edges.append({
                'POINT_1_LNG': p1_lng,
                'POINT_1_LAT': p1_lat,
                'POINT_2_LNG': p2_lng,
                'POINT_2_LAT': p2_lat,
                'DISTANCE_IN_METERS': dist,
                'DURATION_IN_SEC': dur,
                'REPORT_ID': report_id,
                'VELOCITY': velocity
            })
    return edges
preloaded_edges = preload_csv(csv_file)

def build_graph(start_lng, start_lat, end_lng, end_lat):
    """
    构建带有时间加权的图，包括起点和终点到各节点的步行边。
    参数 edges 是从 CSV 预载得到的路线列表。
    """
    G = nx.Graph()

    for edge in preloaded_edges:
        nodeA = (edge['POINT_1_LNG'], edge['POINT_1_LAT'])
        nodeB = (edge['POINT_2_LNG'], edge['POINT_2_LAT'])
        G.add_node(nodeA)
        G.add_node(nodeB)
        G.add_edge(
            nodeA, nodeB,
            distance=edge['DISTANCE_IN_METERS'],
            duration=edge['DURATION_IN_SEC'],
            report_id=edge['REPORT_ID'],
            POINT_1_LNG=edge['POINT_1_LNG'],
            POINT_1_LAT=edge['POINT_1_LAT'],
            POINT_2_LNG=edge['POINT_2_LNG'],
            POINT_2_LAT=edge['POINT_2_LAT'],
            velocity=edge['VELOCITY']
        )

    start_node = (start_lng, start_lat)
    end_node = ( end_lng, end_lat)
    G.add_node(start_node)
    G.add_node(end_node)

    walking_speed = 2.0  # m/s

    for node in G.nodes:
        if node == start_node or node == end_node:
            continue

        if isinstance(node[0], float) and isinstance(node[1], float):
            dist_start_to_node = haversine_distance(start_lng, start_lat, node[0], node[1])
            dur_start_to_node = dist_start_to_node / walking_speed
            G.add_edge(start_node, node, distance=dist_start_to_node, duration=dur_start_to_node, report_id="WALKING", velocity=walking_speed)

            dist_end_to_node = haversine_distance(end_lng, end_lat, node[0], node[1])
            dur_end_to_node = dist_end_to_node / walking_speed
            G.add_edge(end_node, node, distance=dist_end_to_node, duration=dur_end_to_node, report_id="WALKING", velocity=walking_speed)

    dist_start_to_end = haversine_distance(start_lng, start_lat, end_lng, end_lat)
    dur_start_to_end = dist_start_to_end / walking_speed
    G.add_edge(start_node, end_node, distance=dist_start_to_end, duration=dur_start_to_end, report_id="WALKING", velocity=walking_speed)

    return G, start_node, end_node

def get_top_k_fastest_paths(G, start_node, end_node, k=3):
    """
    使用 networkx.shortest_simple_paths 按照 duration 求最短路径，返回前 k 条。
    """
    all_paths = nx.shortest_simple_paths(G, start_node, end_node, weight='duration')

    results = []
    count = 0
    for path in all_paths:
        edges_info = []
        total_duration = 0.0
        total_distance = 0.0

        for i in range(len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            edge_data = G[n1][n2]
            edges_info.append({
                "REPORT_ID": edge_data['report_id'],
                "DURATION_IN_SEC": edge_data['duration'],
                "POINT_1_LNG": edge_data.get('POINT_1_LNG', n1[0]),
                "POINT_1_LAT": edge_data.get('POINT_1_LAT', n1[1]),
                "POINT_2_LNG": edge_data.get('POINT_2_LNG', n2[0]),
                "POINT_2_LAT": edge_data.get('POINT_2_LAT', n2[1]),
                "VELOCITY": edge_data['velocity']
            })
            total_duration += edge_data['duration']
            total_distance += edge_data['distance']

        average_velocity = total_distance / total_duration if total_duration > 0 else 0
        results.append({
            "path_edges": edges_info,
            "total_duration": total_duration,
            "total_distance": total_distance,
            "average_velocity": average_velocity
        })

        count += 1
        if count >= k:
            break

    return results

def plan_routes_function(start_lng, start_lat, end_lng, end_lat, k=3):
    """
    主函数：用已加载好的 edges 来构建图，再找最快的 k 条路径并返回。
    edges: preload_csv(...) 函数返回的路径列表
    """
    G, start_node, end_node = build_graph(start_lng, start_lat, end_lng, end_lat)
    top_paths = get_top_k_fastest_paths(G, start_node, end_node, k)

    routes_output = []
    for i, path_info in enumerate(top_paths, start=1):
        route = {
            "route_rank": i,
            "path": path_info["path_edges"],
            "total_duration": path_info["total_duration"],
            "total_distance": path_info["total_distance"],
            "average_velocity": path_info["average_velocity"]
        }
        routes_output.append(route)

    return round_json_values(routes_output)

# --------------------------
# 使用示例

# 使用示例

# --------------------------
# start_longitude, start_latitude = 10.21284, 56.16184  # 北京天安门
# end_longitude, end_latitude = 10.164431,56.130402 # 北京某地示例
# #
# best_3_routes = plan_routes( start_longitude, start_latitude, end_longitude, end_latitude, k=3)
# # 使用示例
#
# print(best_3_routes)



# --------------------------
# 使用示例

    # 这里示例一个 CSV 文件以及起终点经纬度


# 打印结果


    # for route_info in best_3_routes:
    #     print(f"第{route_info['route_rank']}条路线：")
    #     print("子线路：")
    #     for edge in route_info["path"]:
    #         print(f"    REPORT_ID={edge['REPORT_ID']}, DURATION_IN_SEC={edge['DURATION_IN_SEC']:.1f}")
    #     print(f"总用时：{route_info['total_duration']:.1f} 秒\n")

