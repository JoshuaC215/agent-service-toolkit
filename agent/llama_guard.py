from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from typing import List
from enum import Enum
from pydantic import BaseModel, Field


class SafetyAssessment(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"


class LlamaGuardOutput(BaseModel):
    safety_assessment: SafetyAssessment = Field(description="The safety assessment of the content.")
    unsafe_categories: List[str] = Field(description="If content is unsafe, the list of unsafe categories.", default=[])


unsafe_content_categories = {
    "S1": "Violent Crimes.",
    "S2": "Non-Violent Crimes.",
    "S3": "Sex Crimes.",
    "S4": "Child Exploitation.",
    "S5": "Defamation.",
    "S6": "Specialized Advice.",
    "S7": "Privacy.",
    "S8": "Intellectual Property.",
    "S9": "Indiscriminate Weapons.",
    "S10": "Hate.",
    "S11": "Self-Harm.",
    "S12": "Sexual Content.",
    "S13": "Elections.",
    "S14": "Code Interpreter Abuse."
}

categories_str = "\n".join([f"{k}: {v}" for k, v in unsafe_content_categories.items()])

llama_guard_instructions = f"""
[INST] Task: Check if there is unsafe content in '{{role}}' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{categories_str}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{{conversation_history}}

<END CONVERSATION>

Provide your safety assessment for {{role}} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""


llama_guard_prompt = PromptTemplate.from_template(llama_guard_instructions)
model = ChatGroq(model="llama-guard-3-8b", temperature=0.0)

# Alternate version running on Replicate, also slow :|
# from langchain_community.llms.replicate import Replicate
# model = Replicate(model="meta/meta-llama-guard-2-8b:b063023ee937f28e922982abdbf97b041ffe34ad3b35a53d33e1d74bb19b36c4")


def parse_llama_guard_output(output: str) -> LlamaGuardOutput:
    if output == "safe":
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
    parsed_output = output.split("\n")
    if len(parsed_output) != 2 or parsed_output[0] != "unsafe":
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.ERROR)
    try:
        categories = parsed_output[1].split(",")
        readable_categories = [
            unsafe_content_categories[c.strip()].strip(".") for c in categories
        ]
        return LlamaGuardOutput(
            safety_assessment=SafetyAssessment.UNSAFE,
            unsafe_categories=readable_categories,
        )
    except KeyError:
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.ERROR)


async def llama_guard(role: str, messages: List[AnyMessage]) -> LlamaGuardOutput:
    role_mapping = {"ai": "Agent", "human": "User"}
    messages_str = [f"{role_mapping[m.type]}: {m.content}" for m in messages if m.type in ["ai", "human"]]
    conversation_history = "\n\n".join(messages_str)
    compiled_prompt = llama_guard_prompt.format(role=role, conversation_history=conversation_history)
    result = await model.ainvoke([SystemMessage(content=compiled_prompt)])
    return parse_llama_guard_output(result.content)


if __name__ == "__main__":
    import asyncio

    async def main():
        output = await llama_guard("Agent", [
            HumanMessage(content="Tell me a fun fact?"),
            AIMessage(content="Did you know that honey never spoils?"),
        ])
        print(output)
    asyncio.run(main())
