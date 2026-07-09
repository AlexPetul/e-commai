import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")

db = DocBin()

training_data = [
    (
        "I'm looking for a lawnmower but don't know how to choose",
        {"entities": [(6, 19, "PRODUCT")]},
    )
]

for text, annotations in training_data:
    doc = nlp.make_doc(text)
    ents = []

    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label)

        if span is not None:
            ents.append(span)

    doc.ents = ents
    db.add(doc)

db.to_disk("train.spacy")