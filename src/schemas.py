from dataclasses import dataclass, asdict


@dataclass
class Document:
    id: str
    text: str


@dataclass
class Entity:
    id: str
    text: str
    type: str


@dataclass
class Relation:
    head: str
    tail: str
    type: str
    evidence: str


@dataclass
class Prediction:
    document_id: str
    text: str
    entities: list[Entity]
    relations: list[Relation]

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "text": self.text,
            "entities": [asdict(e) for e in self.entities],
            "relations": [asdict(r) for r in self.relations],
        }
