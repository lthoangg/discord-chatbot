import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")
def get_ner(text = None):
    """
    ner("Hi, my name is Hoang") -> [{"Person": "Hoang"}, ]

    "NORP" : country
    "GPE": city
    "PERSON": person
    "DATE": date
    """

    # text = """Hi, my name is Hoang"""
    doc = nlp(text)

    # print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    # print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

    return [{f"{entity.label_}": entity.text} for entity in doc.ents]

if __name__ == "__main__":
    while True:
        user = input("You: ")
        if user == "quit":
            break
        print(get_ner(user))