import requests
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union

class Job(BaseModel):
    functionName: str = Field(
        description="要调用的功能名称。"
    )
    params: Dict[str, Any] = Field(
        description="功能所需的参数。",
        default={}
    )

class DeviceData(BaseModel):
    device_code: str = Field(
        description="设备代码。",
        examples=["2ea09832"]
    )
    job_list: List[Job] = Field(
        description="要执行的任务列表。",
        default_factory=list
    )
    state: int = Field(
        description="设备状态。",
        default=1,
        examples=[1]
    )

    def send_request(self, api_url: str) -> str:
        """发送 POST 请求到指定的 API。"""
        response = requests.post(api_url, json=self.dict())
        response.raise_for_status()  # 抛出请求错误
        return response.text

class WxInit(Job):
    functionName: str = Field(default="wxInit")
    params: Dict[str, Any] = Field(
        description="微信初始化",
        default={}
    )

class WxHome(Job):
    functionName: str = Field(default="wxHome")
    params: Dict[str, Any] = Field(
        description="微信返回主页",
        default={}
    )

class WxKillApp(Job):
    functionName: str = Field(default="wxKillApp")
    params: Dict[str, Any] = Field(
        description="微信关闭",
        default={}
    )

class WxSendMessageUser(Job):
    functionName: str = Field(default="wxSendMessageUser")
    params: Dict[str, Union[str, None]] = Field(
        description="微信发送消息到联系人",
        default={"phone": None, "content": None},
        examples=[{"phone": '18888888888', "content": "Hello world"}]
    )

class WxSendMessageGroup(Job):
    functionName: str = Field(default="wxSendMessageGroup")
    params: Dict[str, Union[str, None]] = Field(
        description="微信发送消息到群聊",
        default={"groupName": None, "content": None, "mention": None},
        examples=[{"groupName": "欢乐一家人",
                   "content": "Hello world",
                   "mention": "小李"}
                  ]
    )

class WxSetUserInfo(Job):
    functionName: str = Field(default="wxSetUserInfo")
    params: Dict[str, Union[str, None]] = Field(
        description="微信设置用户消息",
        default={"phone": None, "remarks": None, "lablel": None, "addPhone": None, "describe": None},
        examples=[{"phone": "18888888888",
                   "remarks": "张三",
                   "label": "重要客户",
                   "addPhone": "18888888888",
                   "describe": "多次下单的客户"}
                  ]
    )

class WxMoments(Job):
    functionName: str = Field(default="wxMoments")
    params: Dict[str, Union[str, None]] = Field(
        description="微信朋友圈点赞评论",
        default={"phone": None, "sendDate": None, "like": None, "comment": None},
        examples=[{"phone": "18888888888",
                   "sendDate": "5月20日",
                   "like": "true",
                   "comment": "评论内容"}
                  ]
    )

class WxSendMoments(Job):
    functionName: str = Field(default="wxSendMoments")
    params: Dict[str, Any] = Field(
        description="微信发朋友圈",
        default={"photos": [], "content": None},
        examples=[{"photos": ['https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1oHqjl.img',
                              'https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1oHqjl.img'],
                   "content": "Hello world"}
                  ]
    )

class WxAddGroupMember(Job):
    functionName: str = Field(default="wxAddGroupMember")
    params: Dict[str, Union[str, None]] = Field(
        description="微信群聊添加成员",
        default={"groupName": None, "phone": None},
        examples=[{"groupName": "相亲相爱一家人",
                   "phone": "18888888888"}
                  ]
    )

class WxDeleGroupMember(Job):
    functionName: str = Field(default="wxDeleGroupMember")
    params: Dict[str, Union[str, None]] = Field(
        description="微信群聊删除成员",
        default={"groupName": None, "wxAccount": None},
        examples=[{"groupName": "相亲相爱一家人",
                   "wxAccount": "18888888888"}
                  ]
    )

class WxSetGroupNotice(Job):
    functionName: str = Field(default="wxSetGroupNotice")
    params: Dict[str, Union[str, None]] = Field(
        description="微信群聊设置公告",
        default={"groupName": None, "content": None},
        examples=[{"groupName": "相亲相爱一家人",
                   "content": "奉天承运皇帝诏曰"}
                  ]
    )

class WxCallPhone(Job):
    functionName: str = Field(default="wxCallPhone")
    params: Dict[str, Union[str, None]] = Field(
        description="微信语音提醒",
        default={"phone": None},
        examples=[{"phone": "18888888888"}]
    )

if __name__ == "__main__":
    api_url = "http://101.43.11.104:8000/api/job/add_job"
    device_code = "2ea09832"
    wx_add_group_member = WxAddGroupMember(
        params={
            "phone": "a1340430615",
            "groupName": "相亲相爱一家人",
        }
    )
    device_data = DeviceData(
        device_code=device_code,
        job_list=[wx_add_group_member]
    )
    print(device_data)
    try:
        response_text = device_data.send_request(api_url)
        print("响应:", response_text)
    except requests.RequestException as e:
        print(f"请求失败: {e}")