from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class TrainRequest(BaseModel):
    dataset_path: str = Field(..., description="Local folder containing audio files and labels.json")

class StandardResponse(BaseModel):
    message: str = Field(..., description="Result summary message")
    success: bool = Field(..., description="Indicates if operation succeeded")
    data: Optional[Any] = Field(default=None, description="Pay-load data")

class EvaluateData(BaseModel):
    preprocessing_flags: List[str]
    duration_ms: float
    features: Dict[str, Any]
    scores: Dict[str, Any]

class EvaluateResponse(BaseModel):
    message: str
    success: bool
    data: EvaluateData
