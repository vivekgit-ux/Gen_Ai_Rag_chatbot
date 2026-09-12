from typing import Literal
from pydantic import BaseModel, Field


class TriageOutput(BaseModel):
  is_safe: bool = Field(
      ..., description="False agar query unsafe ya injection ho."
  )
  route: Literal["ingest", "retrieve", "direct_chat", "blocked"] = Field(
      ...,
      description=(
          "Agar user file index/load karne ko kahe to 'ingest'; question ho to"
          " 'retrieve'; normal baat ho to 'direct_chat'; unsafe ho to 'blocked'"
      ),
  )
  extracted_file_path: str = Field(
      default="",
      description=(
          "Agar user ne kisi file ka naam ya path diya ho to yahan extract karein"
      ),
  )
  reason: str = Field(..., description="Route chunne ka kaaran")


class RelevanceScore(BaseModel):
  binary_score: Literal["yes", "no"] = Field(
      ..., description="'yes' agar context kaam ka hai, varna 'no'"
  )


class HallucinationScore(BaseModel):
  binary_score: Literal["yes", "no"] = Field(
      ..., description="'yes' agar answer context par aadharit hai, varna 'no'"
  )