from fuzzywuzzy import fuzz, process
from entity_extractor.model import Address
from dataclasses import dataclass
@dataclass
class SearchFeild:
    field_name: str
    value: str

    def __repr__(self):
        return f"SearchFeild(field='{self.field_name}', value='{self.value}')"



# ------------------------------------------
# Step 1: Extract non-empty fields (LIST SAFE)
# ------------------------------------------
def get_non_empty_fields(address: Address) -> dict[str, list[str]]:
    non_empty = {}
    for field, values in address.model_dump().items():
        # values are always lists
        if isinstance(values, list) and len(values) > 0:
            non_empty[field] = values
    return non_empty


# ------------------------------------------------------------------
# Step 2: Iterate over SearchFeild for each NON-EMPTY LIST VALUE
# ------------------------------------------------------------------
def search_address_fields(address: Address, non_empty_fields: dict[str, list[str]]):
    results = []

    for field_name, value_list in non_empty_fields.items():
        for value in value_list:  # create one SearchFeild per value
            sf = SearchFeild(field_name, value)
            results.append(sf)

    return results


# ---------------------------------------------------------
# Step 3: Fuzzy match that handles MULTIPLE QUERY VALUES
# ---------------------------------------------------------
from fuzzywuzzy import process, fuzz

def fuzzy_match_address(query_values: list[str], candidate_values: list[str],field_name:str, threshold=98):
    """
    query_values: list of extracted values from LLM (e.g. ["Green meadows", "Greenmeadows"])
    candidate_values: values from Qdrant metadata

    Returns:
        best_match_value, score, original_query_value
        OR
        (None, 0, None) if no score >= threshold
    """
    if field_name == "region":
        print(candidate_values)
        

    best_global_match = None
    best_global_score = 0
    best_from_query = None

    for qv in query_values:
        match, score = process.extractOne(
            qv,
            candidate_values,
            scorer=fuzz.WRatio  # BEST overall fuzzy scorer
        )

        if score > best_global_score:
            best_global_score = score
            best_global_match = match
            best_from_query = qv

    if best_global_score >= 98:
        return best_global_match, best_global_score, best_from_query

    return None, 0, None
