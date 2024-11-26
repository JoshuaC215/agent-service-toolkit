from enum import Enum

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from core import get_model, settings
from schema.models import GroqModelName


class SafetyAssessment(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"


class LlamaGuardOutput(BaseModel):
    safety_assessment: SafetyAssessment = Field(description="The safety assessment of the content.")
    unsafe_categories: list[str] = Field(
        description="If content is unsafe, the list of unsafe categories.", default=[]
    )


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
    "S14": "Code Interpreter Abuse.",
}

categories_str = "\n".join([f"{k}: {v}" for k, v in unsafe_content_categories.items()])

llama_guard_instructions = f"""
Task: Check if there is unsafe content in '{{role}}' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{categories_str}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{{conversation_history}}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST {{role}} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories."""


def parse_llama_guard_output(output: str) -> LlamaGuardOutput:
    if output == "safe":
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
    parsed_output = output.split("\n")
    if len(parsed_output) != 2 or parsed_output[0] != "unsafe":
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.ERROR)
    try:
        categories = parsed_output[1].split(",")
        readable_categories = [unsafe_content_categories[c.strip()].strip(".") for c in categories]
        return LlamaGuardOutput(
            safety_assessment=SafetyAssessment.UNSAFE,
            unsafe_categories=readable_categories,
        )
    except KeyError:
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.ERROR)


class LlamaGuard:
    def __init__(self) -> None:
        if settings.GROQ_API_KEY is None:
            print("GROQ_API_KEY not set, skipping LlamaGuard")
            self.model = None
            return
        self.model = get_model(GroqModelName.LLAMA_GUARD_3_8B).with_config(tags=["llama_guard"])
        self.prompt = PromptTemplate.from_template(llama_guard_instructions)

    def _compile_prompt(self, role: str, messages: list[AnyMessage]) -> str:
        role_mapping = {"ai": "Agent", "human": "User"}
        messages_str = [
            f"{role_mapping[m.type]}: {m.content}" for m in messages if m.type in ["ai", "human"]
        ]
        conversation_history = "\n\n".join(messages_str)
        return self.prompt.format(role=role, conversation_history=conversation_history)

    def invoke(self, role: str, messages: list[AnyMessage]) -> LlamaGuardOutput:
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
        compiled_prompt = self._compile_prompt(role, messages)
        result = self.model.invoke([HumanMessage(content=compiled_prompt)])
        return parse_llama_guard_output(result.content)

    async def ainvoke(self, role: str, messages: list[AnyMessage]) -> LlamaGuardOutput:
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
        compiled_prompt = self._compile_prompt(role, messages)
        result = await self.model.ainvoke([HumanMessage(content=compiled_prompt)])
        return parse_llama_guard_output(result.content)


if __name__ == "__main__":
    llama_guard = LlamaGuard()
    output = llama_guard.invoke(
        "Agent",
        [
            HumanMessage(content="What's a good way to harm an animal?"),
            AIMessage(
                content="There are many ways to harm animals, but some include hitting them with a stick, throwing rocks at them, or poisoning them."
            ),
        ],
    )
    print(output)
