# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from langchain_core.prompts import PromptTemplate
from .workflow_schema import WorkflowOutput  # your Pydantic model

logger = logging.getLogger(__name__)


class WorkflowProposalConfig(FunctionBaseConfig, name="workflow_proposal_tool"):
    """
    Configuration for the workflow proposal tool.
    """
    llm_name: LLMRef


@register_function(
    config_type=WorkflowProposalConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def workflow_proposal(config: WorkflowProposalConfig, builder: Builder):
    """
    Propose a step-by-step workflow based on provided text requirements,
    returning a structured WorkflowOutput object.
    """

    # 1. Load LLM
    llm = await builder.get_llm(
        llm_name=config.llm_name,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )

    # 2. Define prompt template
    prompt_template = """Below is the Technical Specification (v1.0) and any regulatory/compliance docs:

### Technical Specification
{text}

### Output Format
Produce a JSON object with keys "steps" and "graph", exactly matching the WorkflowOutput schema.
Each step must include:
- id (string)
- name (string)
- resource ("human"|"robot"|"both")
- human_duration (list of two floats) and/or robot_duration (list of two floats)

Each graph edge must include "from" and "to" step IDs.

Return only the JSON; do not include any explanation.
"""
    prompt = PromptTemplate(
        input_variables=["text"],
        template=prompt_template,
    )

    # 3. Enable structured output
    llm_structured = llm.with_structured_output(WorkflowOutput)

    # 4. Build the chain
    chain = prompt | llm_structured

    async def _inner(text: str) -> WorkflowOutput:
        """
        Args:
            text: the combined technical spec + compliance text to analyze.
        Returns:
            A WorkflowOutput instance parsed from the LLM’s JSON.
        """
        output: WorkflowOutput = await chain.ainvoke(text)
        logger.info("Proposed workflow: %s", output.json())
        return output

    # 5. Register
    yield FunctionInfo.from_fn(
        _inner,
        description="Generates a structured workflow (steps + graph) from input text.",
    )
