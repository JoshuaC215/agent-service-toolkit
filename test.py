import aiohttp
import asyncio

async def get_chat_messages(staff_id):
    url = f"https://crm.jisudonghua.com/workphonenew/chatmessages/getChatmessage?staffId={staff_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()  # 返回 JSON 响应
            else:
                print(f"请求失败：{response.status}")
                return None

# 示例调用
async def main():
    staff_id = "2ea09832"
    chat_messages = await get_chat_messages(staff_id)
    print(chat_messages)

# 运行异步主函数
asyncio.run(main())