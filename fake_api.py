import http.client
import json
import time


def message_template(role,new_info):
    new_dict={'role':role,'content':new_info}
    return new_dict
def api_answer(messages,json_format=False):
    conn = http.client.HTTPSConnection("api.openai-hub.com")
    payload = json.dumps({
        "model": "gpt-4o",
        "messages": messages
    })
    if json_format:
        headers = {
            'Authorization': 'Bearer sk-Nf0kLEmbRPRSFdD8qwlg1e7EHuoJMyaf1Z60Fh0IDLYosBEs',
            'Content-Type': 'json'
        }
    else:

        headers = {
            'Authorization': 'Bearer sk-Nf0kLEmbRPRSFdD8qwlg1e7EHuoJMyaf1Z60Fh0IDLYosBEs',
            'Content-Type': 'application/json'
        }
    # conn.request("POST", "/v1/chat/completions", payload, headers)
    # res = conn.getresponse()
    # data = res.read()
    # result = json.loads(data.decode("utf-8"))

    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY = 2  # 重试间隔（秒）

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            result = json.loads(data.decode("utf-8"))

            print(result)
            return result["choices"][0]["message"]["content"]

            # break  # 成功后退出循环
        except TimeoutError:
            print(f"请求超时，正在重试...（第 {attempt} 次尝试）")
            if attempt == MAX_RETRIES:
                print("达到最大重试次数，操作失败。")
                # 根据需要处理失败情况，比如抛出异常或记录日志
                raise
            time.sleep(RETRY_DELAY)  # 等待一段时间后重试
        finally:
            conn.close()  # 确保连接被关闭
