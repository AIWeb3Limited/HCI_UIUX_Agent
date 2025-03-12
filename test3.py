import ast
import re
import sys
from io import StringIO
from azure_api import *

import pandas as pd
pd.set_option('display.max_rows', 2)  # 设置最大显示 10 行

# 测试代码
code = """
tfyhthdrtg
```python
import pandas as pd

pollution_data = pd.read_csv('uploads/pollutionData204273.csv')

# Load the traffic data
traffic_data = pd.read_csv('uploads/trafficData158324.csv')

# Display the first few rows of each dataframe to understand the timestamp format
pollution_data.head(), traffic_data.head()
```
"""
code2 = """
tfyhthdrtg
```python
# Select relevant columns for the final output
final_json={"123":"asdas"}
print("123")
# Output the final answer in JSON format
final_answer = {"hourly_data": final_json}
print({"final_answer": final_answer})
```
"""

# 执行并模拟 Jupyter 的自动显示
local_vars = {}
# result = execute_and_display(extract_python_code(code), local_vars)
result2 = execute_and_display(extract_python_code(code2), local_vars)
# print(result)
print(result2)