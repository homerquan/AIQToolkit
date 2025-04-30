from pydantic import BaseModel, Field
from typing import List, Literal


class Step(BaseModel):
    id: str = Field(..., description="Unique identifier of the step.")
    name: str = Field(..., description="Descriptive name of this step.")
    resource: Literal["human", "robot", "both"] = Field(
        ..., description="Who performs this step."
    )
    human_duration: List[float] = Field(
        None, description="Estimated human time range in hours."
    )
    robot_duration: List[float] = Field(
        None, description="Estimated robot time range in hours."
    )


class Edge(BaseModel):
    from_: str = Field(..., alias="from", description="Step ID where the edge starts.")
    to: str = Field(..., description="Step ID where the edge ends.")


class WorkflowOutput(BaseModel):
    steps: List[Step] = Field(..., description="List of workflow steps.")
    graph: List[Edge] = Field(..., description="Directed edges between steps.")