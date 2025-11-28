import spacy
import re
from difflib import get_close_matches

nlp = spacy.load("en_core_web_sm")

def normalize_text(s):
    return s.strip().upper().replace(".", "").replace(",", "")

def fuzzy_merge(items, cutoff=0.8):
    """Merge similar items using fuzzy matching."""
    merged = []
    for item in items:
        norm = normalize_text(item)
        match = get_close_matches(norm, [normalize_text(m) for m in merged], n=1, cutoff=cutoff)
        if match:
            # Keep existing merged version
            continue
        else:
            merged.append(item)
    return merged

def extract_address_components_with_fuzzy(text: str):
    doc = nlp(text)

    streets = set()
    localities = set()
    towns = set()
    countries = set()
    postcodes = set()

    # --- 1. Extract location entities via SpaCy ---
    location_labels = {"GPE", "LOC", "FAC", "ORG"}
    detected_locations = []
    for ent in doc.ents:
        if ent.label_ in location_labels:
            detected_locations.append(ent.text.strip())

    # --- 2. Extract street addresses using regex ---
    street_pattern = r'\d+\s+[A-Za-z0-9\s]+(?:Rd|Road|St|Street|Ave|Avenue|Lane|Ln|Drive|Dr)\b'
    streets_found = re.findall(street_pattern, text)
    streets.update([s.strip() for s in streets_found])

    # --- 3. Extract postcodes ---
    postcode_pattern = r'\b\d{4}\b'
    postcodes_found = re.findall(postcode_pattern, text)
    postcodes.update([p.strip() for p in postcodes_found])

    # --- 4. Extract countries ---
    country_pattern = r'\b(NZ|New Zealand)\b'
    countries_found = re.findall(country_pattern, text, flags=re.IGNORECASE)
    for c in countries_found:
        countries.add("New Zealand" if c.upper() == "NZ" else c.title())

    # --- 5. Heuristic separation: towns vs localities ---
    streets_set = streets
    remaining_locs = set(detected_locations) - streets_set

    for loc in remaining_locs:
        if len(loc.split()) == 1:
            towns.add(loc)
        else:
            localities.add(loc)

    # --- 6. Fuzzy merge similar entities ---
    streets = fuzzy_merge(streets)
    localities = fuzzy_merge(localities)
    towns = fuzzy_merge(towns)
    countries = fuzzy_merge(countries)
    postcodes = fuzzy_merge(postcodes)

    return {
        "streets": sorted(streets),
        "localities": sorted(localities),
        "towns": sorted(towns),
        "countries": sorted(countries),
        "postcodes": sorted(postcodes)
    }


