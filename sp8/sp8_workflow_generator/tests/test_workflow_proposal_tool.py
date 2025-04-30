# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import pytest
from workflow_proposal_tool import workflow_proposal, WorkflowProposalConfig
from aiq.data_models.component_ref import LLMRef

class DummyAgent:
    def __init__(self, *args, **kwargs):
        pass

    async def arun(self, text, stream=False):
        class Response:
            content = '{"steps":[{"id":"1","name":"TestStep","resource":"human","human_duration":[1,2]}],"graph":[]}'
        return Response()

@pytest.fixture(autouse=True)
def patch_agent_and_env(monkeypatch):
    # Ensure API key for AGNO agents
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    # Stub out AGNO Agent
    monkeypatch.setattr("workflow_proposal_tool.Agent", DummyAgent)

class DummyBuilder:
    async def get_llm(self, llm_name, wrapper_type):
        # Return a placeholder LLM (not used by DummyAgent)
        return None

@pytest.mark.asyncio
async def test_workflow_proposal_generates_expected_json():
    builder = DummyBuilder()
    config = WorkflowProposalConfig(llm_name=LLMRef(id="dummy-llm"))

    # Instantiate the tool
    gen = workflow_proposal(config, builder)
    finfo = await gen.__anext__()

    # Run the inner function
    result = await finfo.fn("dummy specification and compliance text")

    # Assert JSON structure and dummy content
    assert "\"steps\"" in result
    assert "\"graph\"" in result
    assert "TestStep" in result

@pytest.mark.asyncio
async def test_workflow_proposal_raises_without_api_key(monkeypatch):
    # Remove API key to trigger error
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    builder = DummyBuilder()
    config = WorkflowProposalConfig(llm_name=LLMRef(id="dummy-llm"))

    with pytest.raises(ValueError) as excinfo:
        gen = workflow_proposal(config, builder)
        # Attempt to advance generator to trigger API key check
        await gen.__anext__()

    assert "NVIDIA_API_KEY must be set" in str(excinfo.value)
