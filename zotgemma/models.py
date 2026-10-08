"""Plain record types shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field


def zotero_link(key: str, group_id: int | None = None) -> str:
    """``zotero://select`` deep link for an item in the user library (``group_id`` None) or a group."""
    if group_id is None:
        return f"zotero://select/library/items/{key}"
    return f"zotero://select/groups/{group_id}/items/{key}"


@dataclass(slots=True)
class Library:
    """A row of Zotero's ``libraries`` table (joined with ``groups``)."""

    library_id: int
    type: str  # user | group | feed
    group_id: int | None = None
    name: str = ""


@dataclass(slots=True)
class Attachment:
    """An attachment (PDF, HTML snapshot, EPUB, ...) whose text Zotero extracted into storage."""

    item_id: int
    key: str
    parent_item_id: int | None
    filename: str | None
    path: str | None  # absolute path of the file, if resolvable
    content_type: str = ""
    library_id: int = 1


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
    attachment: Attachment | None = None  # best attachment: PDFs first, then the largest text cache
    attachments: list[Attachment] = field(default_factory=list)  # every non-deleted PDF or attachment with a text cache
    standalone: bool = False
    library_id: int = 1
    group_id: int | None = None  # None for the user library
    library_name: str = ""
    extra_citekey: str = ""  # from a "Citation Key:" line in the extra field


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
    ranks: dict[str, int]  # per-source 1-based rank, e.g. {"dense": 3, "keyword": 1, "meta": 2, "fulltext": 1}
    library_id: int = 1
    group_id: int | None = None
    library_name: str = ""

    @property
    def link(self) -> str:
        """The zotero:// deep link that selects this item in the Zotero UI."""
        return zotero_link(self.key, self.group_id)

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
            "library": self.library_name or ("user" if self.group_id is None else f"group {self.group_id}"),
            "link": self.link,
        }
