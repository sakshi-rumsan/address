from pydantic import BaseModel, Field
from typing import List


class Address(BaseModel):

    house_low: List[str] = Field(default_factory=list)
    locality: List[str] = Field(default_factory=list)
    town: List[str] = Field(default_factory=list)
    postcode: List[str] = Field(default_factory=list)
    region: List[str] = Field(default_factory=list)
