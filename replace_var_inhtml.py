import json
import re

# 示例 HTML 代码
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nearby Parking Spots</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f8f9fa;
        }

        .container {
            width: 90%;
            max-width: 500px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .title {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
            padding: 5px;
            background-color: #007bff;
            color: white;
        }

        #map {
            height: 300px;
            width: 100%;
        }

        .button-container {
            padding: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }

        .parking-button {
            padding: 8px 12px;
            font-size: 12px;
            border: none;
            border-radius: 5px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .parking-button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .parking-button.active {
            background-color: #28a745;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="title">Nearby Parking Spots</div>
    <div id="map"></div>
    <div class="button-container" id="button-container"></div>
</div>

<script>

    const searched_result = [
        {
        'reference_point_name': 'Reference Point',
        'reference_point_location': {
            'latitude': 56.18945,
            'longitude': 10.111042
        },
        'nearest_locations': [
            {
                'garagecode': 'SCANDCENTER',
                'latitude': 56.1527,
                'longitude': 10.197,
                'occupancy rate': 0.4462609970674487,
                'distance': 6.727072068746666
            },
            {
                'garagecode': 'MAGASIN',
                'latitude': 56.15679,
                'longitude': 10.2049,
                'occupancy rate': 0.3525,
                'distance': 6.871065285947375
            },
            {
                'garagecode': 'BUSGADEHUSET',
                'latitude': 56.15561,
                'longitude': 10.206,
                'occupancy rate': 1.1041958041958042,
                'distance': 6.999054689763466
            },
            {
                'garagecode': 'NORREPORT',
                'latitude': 56.16184,
                'longitude': 10.21284,
                'occupancy rate': 0.0,
                'distance': 7.030396558719592
            },
            {
                'garagecode': 'SALLING',
                'latitude': 56.15441,
                'longitude': 10.20818,
                'occupancy rate': 0.3041558441558441,
                'distance': 7.185286613662889
            }
        ]
    }
    ];
    modified_result=searched_result[0]
    // Initialize map
    const map = L.map('map').setView([modified_result.reference_point_location.latitude, modified_result.reference_point_location.longitude], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19
    }).addTo(map);

    // Add reference point marker
    const referenceMarker = L.marker([modified_result.reference_point_location.latitude, modified_result.reference_point_location.longitude])
        .addTo(map)
        .bindPopup(`${modified_result.reference_point_name}`)
        .openPopup();

    // Add parking markers
    const parkingMarkers = [];
    modified_result.nearest_locations.forEach(location => {
        const marker = L.marker([location.latitude, location.longitude]).addTo(map)
            .bindPopup(`${location.garagecode}<br>Occupancy Rate: ${(location['occupancy rate'] * 100).toFixed(2)}%`);
        parkingMarkers.push(marker);
    });

    // Create buttons
    const buttonContainer = document.getElementById('button-container');
    modified_result.nearest_locations.forEach((location, index) => {
        const button = document.createElement('button');
        button.classList.add('parking-button');
        button.textContent = `${location.garagecode} (${(location['occupancy rate'] * 100).toFixed(2)}%)`;
        button.addEventListener('click', () => {
            map.setView([location.latitude, location.longitude], 16);
            parkingMarkers.forEach((marker, i) => {
                if (i === index) {
                    marker.openPopup();
                } else {
                    marker.closePopup();
                }
            });
        });
        buttonContainer.appendChild(button);
    });
</script>

</body>
</html>

"""


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
        pattern = rf'(?:const|var|let)\s+{variable_name}\s*=\s*\[.*?\];'
        html_content = re.sub(pattern, lambda m: replacement, html_content, flags=re.DOTALL)

    return html_content
#
aaa=[
        {
        'reference_point_name': 'Rasdasdasd',
        'reference_point_location': {
            'latitude': 56.18945,
            'longitude': 10.111042
        },
        'nearest_locations': [
            {
                'garagecode': 'SCANDCENTER',
                'latitude': 56.1527,
                'longitude': 10.197,
                'occupancy rate': 0.4462609970674487,
                'distance': 6.727072068746666
            },
            {
                'garagecode': 'MAGASIN',
                'latitude': 56.15679,
                'longitude': 10.2049,
                'occupancy rate': 0.3525,
                'distance': 6.871065285947375
            },
            {
                'garagecode': 'BUSGADEHUSET',
                'latitude': 56.15561,
                'longitude': 10.206,
                'occupancy rate': 1.1041958041958042,
                'distance': 6.999054689763466
            },
            {
                'garagecode': 'NORREPORT',
                'latitude': 56.16184,
                'longitude': 10.21284,
                'occupancy rate': 0.0,
                'distance': 7.030396558719592
            },
            {
                'garagecode': 'SALLING',
                'latitude': 56.15441,
                'longitude': 10.20818,
                'occupancy rate': 0.3041558441558441,
                'distance': 7.185286613662889
            }
        ]
    }
    ];
# # #
# replacements_json = json.dumps([{'variable': 'searched_result', 'value': aaa}] )
# # #
# # # # 调用函数
# updated_html = update_js_arrays(html_content, replacements_json)
# # #
# # # # 输出结果
# # print("更新后的HTML:")
# print(updated_html)