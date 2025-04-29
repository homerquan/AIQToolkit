import logging

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

from .file_ingest_function import pdf_file_ingest_tool

logger = logging.getLogger(__name__)

class ComplianceToolConfig(FunctionBaseConfig, name="compliance_checker_tool"):
    pass


@register_function(config_type=ComplianceToolConfig)
async def compliance_checker_tool(config: ComplianceToolConfig, builder: Builder):

    async def _compliance_checker_tool(text: str) -> str:
        compliance_keywords = ["regulation", "compliance", "policy", "law", "governance"]
        if any(keyword in text.lower() for keyword in compliance_keywords):
            return "This is a compliance text"
        return "This is not a compliance text"

    # Create a Generic AgentIQ tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _compliance_checker_tool,
        description=(
            "This tool determines if a given text is related to regulation or compliance. "
            "It analyzes the input text and checks for keywords associated with compliance."
        ),
    )
