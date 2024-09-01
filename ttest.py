# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import requests

def get_chat_messages(staff_id, start_time, end_time):
    url = f"https://crm.jisudonghua.com/workphonenew/chatmessages/getChatmessage"
    params = {
        'staffId': staff_id,
        'start_date': start_time,
        'end_date': end_time
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()  # 返回 JSON 响应
    else:
        print(f"请求失败：{response.status_code}")
        return None

# 示例调用
staff_id = "2ea09832"
# start_time = "2024-08-27T16:20:00Z"  # 示例开始时间
# end_time = "2024-08-27T16:23:00Z"    # 示例结束时间
start_time = "2024-08-27 16:26:00"
end_time = "2024-08-27 16:49:00"
chat_messages = get_chat_messages(staff_id, start_time, end_time)
print(chat_messages)