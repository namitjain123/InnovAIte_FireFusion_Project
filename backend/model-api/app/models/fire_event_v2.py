from typing import List
from pydantic import BaseModel


class TensorShapeV2(BaseModel):
    window_days: int
    steps_per_day: int
    total_timesteps: int
    static_features: int
    temporal_features: int


class StaticFeaturesV2(BaseModel):
    feature_order: List[str]
    values: List[float]


class TemporalTimestepV2(BaseModel):
    day: int
    hour: int
    values: List[float]


class TemporalFeaturesV2(BaseModel):
    feature_order: List[str]
    timesteps: List[TemporalTimestepV2]


class CellInputV2(BaseModel):
    cell_id: str
    static_features: StaticFeaturesV2
    temporal_features: TemporalFeaturesV2


class ModelInferenceInputV2(BaseModel):
    schema_version: str
    request_id: str
    created_at: str

    grid_resolution_m: int
    cell_area_ha: float
    forecast_horizon_h: int

    tensor_shape: TensorShapeV2

    cells: List[CellInputV2]