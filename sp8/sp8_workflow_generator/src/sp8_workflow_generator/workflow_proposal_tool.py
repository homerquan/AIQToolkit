# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.function_info import FunctionInfo
from aiq.data_models.component_ref import LLMRef
from aiq.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


# -----------------------------
# File: workflow_proposal_tool.py
# -----------------------------
class WorkflowProposalConfig(FunctionBaseConfig, name="workflow_proposal_tool"):
    """
    Configuration for the workflow proposal tool.
    """

    llm_name: LLMRef = Field(description="LLM to use for generating workflows.")


@register_function(
    config_type=WorkflowProposalConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def workflow_proposal(config: WorkflowProposalConfig, builder: Builder):
    """
    Propose a step-by-step workflow based on provided text requirements.
    """
    # Load the specified LLM
    llm = await builder.get_llm(
        llm_name=config.llm_name,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )

    async def _inner(text: str) -> str:
        prompt = (
            "Below is the Technical Specification (v1.0) for the AI Agent Analysis component,\n"
            "along with any regulation and compliance documents provided.  \n\n"
            "### Technical Specification\n"
            "Component: AI Agent Analysis – Documents to Workflow Graph\n"
            "Date: 2025-04-07\n"
            "…[insert the full spec here]…\n\n"
            "### Regulatory & Compliance Documents\n"
            "…[insert or reference your compliance texts here]…\n\n"
            "Using **both** the technical spec and the compliance docs, identify every operational step\n"
            "required to meet the stated goal, highlight which steps present an opportunity for **robot**\n"
            "automation or **human-robot collaboration**, and estimate duration ranges (in hours) for\n"
            "each (human_duration and/or robot_duration).  \n\n"
            "**Output only** the JSON object with two keys—`steps` and `graph`—in **exactly** this format:\n"
            "```json\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "id": "1",\n'
            '      "name": "Pre-check",\n'
            '      "resource": "human",\n'
            '      "human_duration": [1, 2]\n'
            "    },\n"
            "    …\n"
            "  ],\n"
            '  "graph": [\n'
            '    { "from": "1", "to": "2" },\n'
            "    …\n"
            "  ]\n"
            "}\n"
            "```"
        )

        response = await llm.ainvoke(prompt)
        return response.content

    yield FunctionInfo.from_fn(
        _inner,
        description="Generates a step-by-step workflow based on the input text.",
    )
