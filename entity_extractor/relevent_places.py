import os
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from fuzzywuzzy import fuzz, process

from entity_extractor.fuzzy_wazzy import fuzzy_match_address, get_non_empty_fields
from entity_extractor.model import Address
from entity_extractor.search_feild import SearchFeilds


# -----------------------------
# Your settings loader



load_dotenv()
class Settings:
    model_name: str =  os.getenv("CHAT_MODEL")
    ollama_url: str = os.getenv("OLLAMA_URL")

def get_settings():
    return Settings()
# -----------------------------
# Structured Address BaseModel (LIST FIELDS)
# -----------------------------

# -----------------------------
# Address Analyzer
# -----------------------------
class AddressAnalyzer:
    """Analyze a user query and return structured address metadata."""

    def __init__(self, model_name: Optional[str] = None, ollama_url: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.model_name
        self.ollama_url = ollama_url or self.settings.ollama_url

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.ollama_url,
            timeout=300
        )

        self.llm_with_tools = self.llm.bind_tools([Address])
        self.output_parser = PydanticToolsParser(tools=[Address])
        self._prompt: Optional[ChatPromptTemplate] = None

    # -----------------------------
    # Create Prompt Template
    # -----------------------------
    def _get_prompt(self) -> ChatPromptTemplate:
        if self._prompt is None:
            system_prompt = """
You are an expert address parser.

IMPORTANT RULES:
- Return **each field as a LIST of strings**, even if only one value exists.
- Unknown values must return an empty list [].
- The output MUST strictly match the Address BaseModel.
"""

            self._prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
        return self._prompt

    # -----------------------------
    # Parse Address
    # -----------------------------
    def parse_address(self, query: str) -> Address:
        prompt = self._get_prompt()
        chain = prompt | self.llm_with_tools | self.output_parser
        result = chain.invoke({"input": query})

        

        # Case 1 — already an Address instance
        if isinstance(result, Address):
            return result

        # Case 2 — dict returned
        if isinstance(result, dict):
            normalized = self._normalize_to_lists(result)
            return Address(**normalized)

        # Case 3 — string JSON
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                normalized = self._normalize_to_lists(parsed)
                return Address(**normalized)
            except:
                return Address()

        # Case 4 — list from tool
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, Address):
                return item
            if isinstance(item, dict):
                return Address(**self._normalize_to_lists(item))

        return result[0]

    # -----------------------------
    # Ensure every value is a List[str]
    # -----------------------------
    def _normalize_to_lists(self, raw: Dict[str, Any]) -> Dict[str, List[str]]:
        normalized = {}
        for key in Address.model_fields.keys():
            val = raw.get(key, [])
            if isinstance(val, str):
                val = [val]
            elif isinstance(val, list):
                val = [str(v) for v in val]
            else:
                val = []
            normalized[key] = val
        return normalized


# -----------------------------
# Helper: Extract non-empty fields
# -----------------------------

# -----------------------------
# Example Usage
# -----------------------------
async def run_workflow():
    analyzer = AddressAnalyzer()

    user_query = (
        "i live on a small flat on 46 Mishon Rd, Green meadows in Napier. "
    
    )

    address_result = analyzer.parse_address(user_query)

    print("\nFinal Parsed Address:")
    print(address_result)

    print("\nNon-empty fields:")
    from dataclasses import dataclass


    print("\n=== STEP 1: NON-EMPTY FIELDS ===")
    non_empty_fields = get_non_empty_fields(address_result)
    print(non_empty_fields)
    qdrant_dummy_candidates ={}
    for x in non_empty_fields:
        y=await SearchFeilds(x)
        qdrant_dummy_candidates[x] =y





    print("\n=== STEP 3: FUZZY MATCHING ===")
    best_matches_overall = {}

    # Loop over non-empty fields and get best match from Qdrant
    from fuzzywuzzy import fuzz, process

    for field_name, values in non_empty_fields.items():
        if field_name in qdrant_dummy_candidates:
            print(f"\nField: {field_name}")
            best_match, score, original = fuzzy_match_address(
                values, 
                qdrant_dummy_candidates[field_name],
                field_name
                
            )
            print("  Original LLM value:", original)
            print("  Best match:", best_match)
            print("  Fuzzy score:", score)
            best_matches_overall[field_name] = best_match
            


    print("\n=== STEP 2: BEST MATCH PER FIELD ===")
    print(best_matches_overall)

    return best_matches_overall

