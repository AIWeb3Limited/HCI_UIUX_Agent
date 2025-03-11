import ast
import sys
from io import StringIO
import pandas as pd


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


# 测试代码
code = """
import pandas as pd
pollution_data = pd.read_csv('uploads\\\\pollutionData204273.csv')
pollution_data['timestamp'] = pd.to_datetime(pollution_data['timestamp'])
pollution_data
"""

# 执行并模拟 Jupyter 的自动显示
local_vars = {}
result = execute_and_display(code, local_vars)
print(result)