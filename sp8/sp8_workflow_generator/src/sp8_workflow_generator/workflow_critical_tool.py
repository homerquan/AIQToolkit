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
# File: workflow_critical_tool.py
# -----------------------------
class WorkflowCriticalConfig(FunctionBaseConfig, name="workflow_critical_tool"):
    """
    Configuration for the workflow critical evaluation tool.
    """
    llm_name: LLMRef = Field(description="LLM to use for critical analysis and compliance checking.")


@register_function(config_type=WorkflowCriticalConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def workflow_critical(config: WorkflowCriticalConfig, builder: Builder):
    """
    Critically evaluate a proposed workflow against regulatory and compliance requirements.
    """
    # Load the specified LLM
    llm = await builder.get_llm(
        llm_name=config.llm_name,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )

    async def _inner(text: str, proposed_workflow: str) -> str:
        prompt = (
            f"Text to analyze for regulations/compliance:\n{text}\n\n"
            f"Proposed Workflow:\n{proposed_workflow}\n\n"
            "Identify any regulatory or compliance requirements present in the text, "
            "critique the proposed workflow for compliance, and suggest improvements "
            "to ensure it adheres to all regulations."
        )
        response = await llm.ainvoke(prompt)
        return response.content

    yield FunctionInfo.from_fn(
        _inner,
        description=(
            "Critically evaluates a proposed workflow against regulatory and compliance "
            "requirements, suggesting improvements to ensure full compliance."
        ),
    )
