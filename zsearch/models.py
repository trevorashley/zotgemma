"""Plain record types shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Attachment:
    """A PDF attachment in Zotero storage."""

    item_id: int
    key: str
    parent_item_id: int | None
    filename: str | None
    path: str | None  # absolute path of the file, if resolvable


@dataclass(slots=True)
class Annotation:
    """A highlight/note annotation on an attachment."""

    item_id: int
    page_label: str | None
    text: str
    comment: str


@dataclass(slots=True)
class Item:
    """A bibliographic item (or standalone PDF) from the Zotero library."""

    item_id: int
    key: str
    item_type: str
    title: str
    year: int | None
    authors: str
    venue: str
    abstract: str
    doi: str
    url: str
    date_modified: str
    tags: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    attachment: Attachment | None = None  # best PDF: the one with the largest text cache
    attachments: list[Attachment] = field(default_factory=list)  # every non-deleted PDF
    standalone: bool = False


@dataclass(slots=True)
class Hit:
    """A ranked search result."""

    item_id: int
    key: str
    citekey: str | None
    title: str
    authors: str
    year: int | None
    item_type: str
    venue: str
    score: float
    ranks: dict[str, int]  # per-source 1-based rank, e.g. {"dense": 3, "bm25": 1}

    @property
    def link(self) -> str:
        """The zotero:// deep link that selects this item in the Zotero UI."""
        return f"zotero://select/library/items/{self.key}"

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "citekey": self.citekey,
            "key": self.key,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "item_type": self.item_type,
            "venue": self.venue,
            "score": round(self.score, 5),
            "ranks": self.ranks,
            "link": self.link,
        }
