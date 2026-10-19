# Source:StonyBrookNLP/musique at 922ac98f19a201998dbdae6d7f2887a5258dbdeb;upstreamattribution/CCBY4.0 applies.
def translate_id(key: str) -> str:
    # Naming convention change for question-ids in the released dataset format.
    namechange = {
        "double": "2hop", "triple_ii": "3hop1",
        "triple_io": "3hop2", "quadruple_iii": "4hop1",
        "quadruple_iot": "4hop2", "quadruple_ioh1": "4hop3", "quadruple_ioh2": "4hop4"
    }
    assert sum([original in key for original, new in namechange.items()]) == 1
    for original, new in namechange.items():
        key = key.replace(original, new)
    return key
