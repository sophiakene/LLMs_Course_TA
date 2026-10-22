"""Solutions: anything that turns prompts into answers.

Every solution implements predict(docs) -> list of answer strings, one per doc.
Add your own by subclassing Solution (or VLLMChat) and registering it in
SOLUTIONS at the bottom of the file.

RULE: the benchmark prompts are the test set. Do not copy them into system
prompts, few-shot examples, retrieval indexes or training data.
"""

MAX_TOKENS = 200  # matches the benchmark; changing it makes scores incomparable


class Solution:
    def predict(self, docs: list[dict]) -> list[str]:
        raise NotImplementedError


class VLLMChat(Solution):
    """Chat model served by vLLM. YOUR BASELINE: implement the three methods.

    SystemPrompt and FewShot below build on this class, so your solutions
    automatically use the same generation settings as your baseline.
    vLLM docs: https://docs.vllm.ai/en/latest/ (look for LLM, SamplingParams, LLM.chat)
    """

    system_prompt: str | None = None

    def __init__(self, model: str, gpu_memory_utilization: float = 0.85):
        from vllm import LLM, SamplingParams

        # TODO: load the model with vLLM and store it as self.llm.
        #   - pass gpu_memory_utilization on to vLLM
        #   - our T4 GPUs do not support bfloat16: use dtype="half"
        # TODO: define the decoding settings as self.params (a SamplingParams).
        #   - decide: greedy or sampling? How long may an answer be (MAX_TOKENS)?
        #   - explain your choices in the report
        raise NotImplementedError

    def build_messages(self, doc: dict) -> list[dict]:
        """Turn one benchmark row into chat messages. Override to change prompting."""
        # TODO: return a list of chat messages, e.g. [{"role": "user", "content": ...}].
        #   - the prompt text is in doc["prompt"]
        #   - if self.system_prompt is set, put it first with role "system"
        #     (some chat templates, e.g. Gemma's, reject a system role)
        raise NotImplementedError

    def predict(self, docs: list[dict]) -> list[str]:
        # TODO: return one answer string per doc, in the same order as docs.
        #   - use build_messages for every doc
        #   - send ALL conversations to vLLM in one call; one call per prompt
        #     takes hours instead of minutes
        #   - LLM.chat applies the model's chat template; LLM.generate does not
        raise NotImplementedError


class Baseline(VLLMChat):
    """The model as-is. Every intervention is compared against this."""


class SystemPrompt(VLLMChat):
    """Intervention 1: steer the model with a system prompt."""

    # TODO: write your own prompt and state your hypothesis: should it change
    # U (unanchored disagreement), A (anchored disagreement) or both?
    system_prompt = (
        "Answer as a typical person from the culture associated with the "
        "language you are addressed in."
    )


class FewShot(VLLMChat):
    """Intervention 2: show example question/answer pairs before the real prompt."""

    # TODO: write examples in each evaluation language, using your OWN questions.
    # {language code: [(question, answer), ...]}
    examples: dict[str, list[tuple[str, str]]] = {
        "en": [("What do people usually eat for lunch on a workday? Answer in one sentence.",
                "TODO: a culturally specific answer")],
    }

    def build_messages(self, doc: dict) -> list[dict]:
        messages = []
        for question, answer in self.examples.get(doc["language"], []):
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": doc["prompt"]})
        return messages


class TranslatePipeline(Solution):
    """Intervention 3: translate to English, answer, translate back.

    Hypothesis to test: this removes the language signal, so U should drop.
    """

    def __init__(self, model: str, gpu_memory_utilization: float = 0.85):
        # TODO: load an answering model and a translation model (e.g. NLLB,
        # or reuse the chat model with a "Translate to ..." instruction).
        raise NotImplementedError

    def predict(self, docs: list[dict]) -> list[str]:
        # TODO:
        # 1. translate every doc["prompt"] into English (batch it!)
        # 2. answer the English prompts
        # 3. translate each answer back into doc["language"]
        raise NotImplementedError


# TODO: further ideas: retrieval over country facts (RAG), a LoRA fine-tuned
# model (vLLM: LLM(model=..., enable_lora=True) + LoRARequest), model merging.


SOLUTIONS = {
    "baseline": Baseline,
    "system_prompt": SystemPrompt,
    "few_shot": FewShot,
    "translate": TranslatePipeline,
}
