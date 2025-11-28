from vector_db.entity_extractor import extract_address_components_with_fuzzy


import spacy
import re

nlp = spacy.load("en_core_web_sm")

def normalize_component(part):
    # Normalize country codes
    if part.upper() == "NZ":
        return "New Zealand"
    return part


# Function to extract lines that may contain addresses
def extract_addresses_linewise(text: str):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    addresses = []
    street_pattern = r'\d+\s+[A-Za-z0-9\s]+(?:Rd|Road|St|Street|Ave|Avenue|Lane|Ln|Drive|Dr)\b'
    postcode_pattern = r'\b\d{4}\b'
    country_pattern = r'\b(NZ|New Zealand)\b'

    for sent in sentences:
        if re.search(street_pattern, sent) or re.search(postcode_pattern, sent) or re.search(country_pattern, sent, flags=re.IGNORECASE):
            addresses.append(sent)
    return addresses

# Function to extract address parts in order from a line
def extract_address_parts(line, parts):
    found = []
    line_lower = line.lower()
    for part in parts:
        for match in re.finditer(re.escape(part.lower()), line_lower):
            found.append((match.start(), part))
    found.sort()
    return [part for _, part in found]


from fuzzywuzzy import fuzz

def extract_address_parts_fuzzy(line, parts, threshold=80):
    matched = []
    for part in parts:
        if fuzz.partial_ratio(part.lower(), line.lower()) >= threshold:
            matched.append(part)
    return matched

def run_workflow(text: str):
    # Step 1: Extract all components
    address_components = extract_address_components_with_fuzzy(text)
    address_parts = sum(address_components.values(), [])
    address_parts = [normalize_component(p) for p in address_parts]

    # Step 2: Split text into address lines
    address_lines = extract_addresses_linewise(text)

    # Step 3: Extract parts from each line using fuzzy match
    extracted_addresses = []
    for line in address_lines:
        parts_in_line = extract_address_parts_fuzzy(line, address_parts)
        if parts_in_line:
            extracted_addresses.append(", ".join(parts_in_line))

    # Step 4: Print and return
    addresses = []
    for i, addr in enumerate(extracted_addresses, 1):
        print(f"Address {i}: {addr}")
        addresses.append(addr)
    return addresses

