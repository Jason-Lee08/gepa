# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from gepa.image import Image
from gepa.proposer.reflective_mutation.base import Signature


class InstructionProposalSignature(Signature):
    default_prompt_template = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

    input_keys: ClassVar[list[str]] = ["current_instruction_doc", "dataset_with_feedback", "prompt_template"]
    output_keys: ClassVar[list[str]] = ["new_instruction"]

    @classmethod
    def validate_prompt_template(cls, prompt_template: str | None) -> None:
        if prompt_template is None:
            return
        missing_placeholders = [
            placeholder for placeholder in ("<curr_param>", "<side_info>") if placeholder not in prompt_template
        ]
        if missing_placeholders:
            raise ValueError(f"Missing placeholder(s) in prompt template: {', '.join(missing_placeholders)}")

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        current_instruction = input_dict.get("current_instruction_doc")
        if not isinstance(current_instruction, str):
            raise TypeError("current_instruction_doc must be a string")

        dataset = input_dict.get("dataset_with_feedback")
        if not isinstance(dataset, Sequence) or isinstance(dataset, str | bytes):
            raise TypeError("dataset_with_feedback must be a sequence of records")

        def format_samples(samples: Sequence[Mapping[str, Any]]) -> tuple[str, list[Image]]:
            """Render samples as markdown, extracting any Image objects.

            Returns:
                A tuple of (formatted_text, collected_images).  Image objects
                are replaced with ``[IMAGE-N]`` placeholders in the text.
            """
            collected_images: list[Image] = []

            def render_value(value: Any, level: int = 3) -> str:
                # level controls markdown header depth (###, ####, etc.)
                if isinstance(value, Image):
                    collected_images.append(value)
                    return f"[IMAGE-{len(collected_images)} — see visual content]\n\n"
                elif isinstance(value, dict):
                    s = ""
                    for k, v in value.items():
                        s += f"{'#' * level} {k}\n"
                        s += render_value(v, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                elif isinstance(value, list | tuple):
                    s = ""
                    for i, item in enumerate(value):
                        s += f"{'#' * level} Item {i + 1}\n"
                        s += render_value(item, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                else:
                    return f"{str(value).strip()}\n\n"

            def convert_sample_to_markdown(sample: Mapping[str, Any], examplenum: int) -> str:
                s = f"# Example {examplenum}\n"
                for key, val in sample.items():
                    s += f"## {key}\n"
                    s += render_value(val, level=3)
                return s

            text = "\n\n".join(convert_sample_to_markdown(sample, i + 1) for i, sample in enumerate(samples))
            return text, collected_images

        prompt_template = input_dict.get("prompt_template")
        if prompt_template is None:
            prompt_template = cls.default_prompt_template

        cls.validate_prompt_template(prompt_template)

        formatted_text, images = format_samples(dataset)

        if images:
            formatted_text = (
                f"The evaluation data below includes visual content ({len(images)} image(s)). "
                "Analyze both the text and images when suggesting improvements.\n\n" + formatted_text
            )

        prompt = prompt_template.replace("<curr_param>", current_instruction)
        prompt = prompt.replace("<side_info>", formatted_text)

        # When images are present, return an OpenAI-compatible multimodal
        # messages list so the reflection LM receives the images inline.
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                content.append(img.to_openai_content_part())
            return [{"role": "user", "content": content}]

        return prompt

    @classmethod
    def run(cls, lm: Any, input_dict: Mapping[str, Any]) -> dict[str, str]:
        full_prompt = cls.prompt_renderer(input_dict)
        if isinstance(full_prompt, list):
            # Multimodal messages list — print the text parts
            for msg in full_prompt:
                for part in msg.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "text":
                        print(part["text"])
                    elif isinstance(part, str):
                        print(part)
        else:
            print(full_prompt)
        print("=" * 80)
        lm_res = lm(full_prompt)
        lm_out = lm_res.strip()
        return cls.output_extractor(lm_out)

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        def extract_instruction_text() -> str:
            # Find the opening ``` fence (first occurrence).
            open_pos = lm_out.find("```")
            if open_pos == -1:
                return lm_out.strip()

            # Check whether there is a *second* ``` anywhere (distinct position).
            # If find and rfind land on the same position, there is only one fence.
            last_pos = lm_out.rfind("```")
            if last_pos == open_pos:
                # Only one ``` in the entire string — treat it as a delimiter and
                # return whichever side has content, matching original behaviour.
                stripped = lm_out.strip()
                if stripped.startswith("```"):
                    m = re.match(r"^```\S*\n?", stripped)
                    if m:
                        return stripped[m.end():].strip()
                elif stripped.endswith("```"):
                    return stripped[:-3].strip()
                return stripped

            after_open = open_pos + 3

            # Skip an optional language / info specifier on the same line as the
            # opening fence (e.g. "python", "7", "text").  We only skip it when
            # the rest of that line has NO spaces — so that lines of actual content
            # that happen to start right after ``` aren't eaten.
            content_start = after_open
            first_nl = lm_out.find("\n", after_open)
            if first_nl != -1:
                specifier = lm_out[after_open:first_nl]
                if specifier and " " not in specifier:
                    content_start = first_nl + 1

            # Find the closing fence: the LAST ``` that sits alone on its own line
            # (i.e. preceded by \n and followed only by optional whitespace or
            # end-of-string).  This pattern avoids matching ``` that open or close
            # code blocks *inside* the proposed system prompt content.
            closing_fence_re = re.compile(r"\n```[ \t]*(?:\n|$)")
            close_match = None
            for m in closing_fence_re.finditer(lm_out, content_start):
                close_match = m  # keep the last match

            if close_match is not None:
                # content runs from content_start up to the \n that begins the fence
                content = lm_out[content_start : close_match.start()]
                return content.strip()

            # Fallback: no standalone closing fence found (e.g. truncated response).
            # Use rfind to find the last ``` and treat it as the closing fence.
            end = lm_out.rfind("```")
            if end > open_pos:
                content = lm_out[content_start:end]
                return content.strip()

            # No closing fence — return everything after the opening fence.
            return lm_out[content_start:].strip()

        return {"new_instruction": extract_instruction_text()}
