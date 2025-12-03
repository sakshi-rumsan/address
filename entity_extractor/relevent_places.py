import os
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate

from entity_extractor.fuzzy_wuzzy import fuzzy_match_address, get_non_empty_fields
from entity_extractor.model import Address
from entity_extractor.search_field import SearchFeilds, search_qdrant_by_filter

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


class Settings:
    model_name: str = os.getenv("CHAT_MODEL")
    ollama_url: str = os.getenv("OLLAMA_URL")


def get_settings():
    return Settings()


# -----------------------------
# Address Analyzer
# -----------------------------
class AddressAnalyzer:
    """Analyze a user query and return structured address metadata."""

    def __init__(
        self, model_name: Optional[str] = None, ollama_url: Optional[str] = None
    ):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.model_name
        self.ollama_url = ollama_url or self.settings.ollama_url

        self.llm = ChatOllama(
            model=self.model_name, base_url=self.ollama_url, timeout=300
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
- The user may mention MULTIPLE ADDRESSES.
- You MUST output ONE Address tool call per detected address.
- DO NOT merge different people's addresses.
- Return EACH FIELD as a LIST of strings, even if only one value exists.
- Unknown or missing values must return an empty list [].
- The output MUST strictly match the Address BaseModel:
    house_low, house_high, locality, town, postcode, region

Example:
User: "I live at 10 King St, Wellington. My brother lives in Palmerston North."
→ Output 2 separate Address tool calls (JSON array):

[{{"house_low":["10"], "house_high":[], "locality":["King St"], "town":["Wellington"], "postcode":[], "region":[]}},
 {{"house_low":[], "house_high":[], "locality":[], "town":["Palmerston North"], "postcode":[], "region":[]}}]


"""
            self._prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{input}")]
            )
        return self._prompt

    # -----------------------------
    # Parse Address → supports multiple addresses
    # -----------------------------
    def parse_address(self, query: str) -> List[Address]:
        prompt = self._get_prompt()
        chain = prompt | self.llm_with_tools | self.output_parser
        result = chain.invoke({"input": query})
        print("Raw LLM output:", result)

        addresses: List[Address] = []

        # Handle string output (JSON)
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                parsed = result

            if isinstance(parsed, list):
                for item in parsed:
                    addresses.extend(self._split_combined_address(item))
            elif isinstance(parsed, dict):
                addresses.extend(self._split_combined_address(parsed))

        # Handle dict output
        elif isinstance(result, dict):
            addresses.extend(self._split_combined_address(result))

        # Handle Address object
        elif isinstance(result, Address):
            addresses.extend(self._split_combined_address(result.dict()))

        # Handle list of mixed objects
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, Address):
                    addresses.extend(self._split_combined_address(item.dict()))
                elif isinstance(item, dict):
                    addresses.extend(self._split_combined_address(item))

        return addresses

    # -----------------------------
    # Normalize all values to List[str]
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
    # Split combined Address fields into multiple Address objects
    # -----------------------------
    def _split_combined_address(self, raw: Dict[str, Any]) -> List[Address]:
        normalized = self._normalize_to_lists(raw)
        # Determine max number of addresses from the longest field
        max_len = max(len(v) for v in normalized.values())
        split_addresses = []

        for i in range(max_len):
            addr_dict = {}
            for key, values in normalized.items():
                addr_dict[key] = [values[i]] if i < len(values) else []
            split_addresses.append(Address(**addr_dict))
        return split_addresses


# -----------------------------
# Workflow
import os
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate

from entity_extractor.fuzzy_wuzzy import fuzzy_match_address, get_non_empty_fields
from entity_extractor.model import Address
from entity_extractor.search_field import SearchFeilds

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


class Settings:
    model_name: str = os.getenv("CHAT_MODEL")
    ollama_url: str = os.getenv("OLLAMA_URL")


def get_settings():
    return Settings()


# -----------------------------
# Address Analyzer
# -----------------------------
class AddressAnalyzer:
    """Analyze a user query and return structured address metadata."""

    def __init__(
        self, model_name: Optional[str] = None, ollama_url: Optional[str] = None
    ):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.model_name
        self.ollama_url = ollama_url or self.settings.ollama_url

        self.llm = ChatOllama(
            model=self.model_name, base_url=self.ollama_url, timeout=300
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
- The user may mention MULTIPLE ADDRESSES.
- You MUST output ONE Address tool call per detected address.
- DO NOT merge different people's addresses.
- Return EACH FIELD as a LIST of strings, even if only one value exists.
- Unknown or missing values must return an empty list [].
- The output MUST strictly match the Address BaseModel:
    house_low, house_high, locality, town, postcode, region

Example:
User: "I live at 10 King St, Wellington. My brother lives in Palmerston North."
→ Output 2 separate Address tool calls (JSON array):

[{{"house_low":["10"], "house_high":[], "locality":["King St"], "town":["Wellington"], "postcode":[], "region":[]}},
 {{"house_low":[], "house_high":[], "locality":[], "town":["Palmerston North"], "postcode":[], "region":[]}}]
"""
            self._prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{input}")]
            )
        return self._prompt

    # -----------------------------
    # Parse Address → supports multiple addresses
    # -----------------------------
    def parse_address(self, query: str) -> List[Address]:
        prompt = self._get_prompt()
        chain = prompt | self.llm_with_tools | self.output_parser
        result = chain.invoke({"input": query})
        print("Raw LLM output:", result)

        addresses: List[Address] = []

        # Handle string output (JSON)
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                parsed = result

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        addresses.append(Address(**self._normalize_to_lists(item)))
            elif isinstance(parsed, dict):
                addresses.append(Address(**self._normalize_to_lists(parsed)))

        # Handle dict output
        elif isinstance(result, dict):
            addresses.append(Address(**self._normalize_to_lists(result)))

        # Handle Address object
        elif isinstance(result, Address):
            addresses.append(result)

        # Handle list of mixed objects
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, Address):
                    addresses.append(item)
                elif isinstance(item, dict):
                    addresses.append(Address(**self._normalize_to_lists(item)))

        # Split combined fields into separate addresses
        final_addresses = []
        for addr in addresses:
            final_addresses.extend(self._split_combined_address(addr.dict()))

        return final_addresses

    # -----------------------------
    # Normalize all values to List[str]
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
    # Split combined Address fields into multiple Address objects
    # -----------------------------
    def _split_combined_address(self, raw: Dict[str, Any]) -> List[Address]:
        normalized = self._normalize_to_lists(raw)
        max_len = max(len(v) for v in normalized.values())
        split_addresses = []

        for i in range(max_len):
            addr_dict = {
                key: [values[i]] if i < len(values) else []
                for key, values in normalized.items()
            }
            split_addresses.append(Address(**addr_dict))
        return split_addresses


# -----------------------------
# Workflow
# -----------------------------


# -----------------------------
# Workflow
# -----------------------------
async def run_workflow(user_query: str):
    analyzer = AddressAnalyzer()

    # Step 1: Parse Addresses
    address_results = analyzer.parse_address(user_query)
    print("\n=== Parsed Addresses ===")
    for i, addr in enumerate(address_results):
        print(f"Address {i+1}: {addr}")

    # Step 2: Extract Non-empty Fields & Qdrant Dummy Search
    qdrant_candidates_all = []
    for i, addr in enumerate(address_results):
        print(f"\n--- Address {i+1}: Non-empty Fields ---")
        non_empty_fields = get_non_empty_fields(addr)
        print(non_empty_fields)

        # Fetch candidate values for each field
        qdrant_dummy_candidates = {}
        for field_name, values in non_empty_fields.items():
            y = await SearchFeilds(field_name)
            qdrant_dummy_candidates[field_name] = y

        qdrant_candidates_all.append(qdrant_dummy_candidates)

    # Step 3: Fuzzy Matching per address
    merged_best_matches = {}
    for i, addr in enumerate(address_results):
        best_matches = {}
        non_empty_fields = get_non_empty_fields(addr)
        qdrant_candidates = qdrant_candidates_all[i]

        for field_name, values in non_empty_fields.items():
            if field_name in qdrant_candidates:
                best_match, score, original = fuzzy_match_address(
                    values, qdrant_candidates[field_name], field_name
                )
                best_matches[field_name] = {
                    "original": original,
                    "best_match": best_match,
                    "score": score,
                }

        # Keep one entry per parsed address
        merged_best_matches[f"address_{i+1}"] = best_matches

    print(f"\n=== Merged Best Matches ===\n{merged_best_matches}")

    # Step 4: Query Qdrant separately for each address
    final_results = []
    for addr_key, filter_dict in merged_best_matches.items():
        if isinstance(filter_dict, dict):
            res = await search_qdrant_by_filter(filter_dict, query=user_query)
            if isinstance(res, list) and res:
                final_results.append({"address_key": addr_key, "results": res})

    print("\n=== Final Qdrant Results ===")

    return final_results
