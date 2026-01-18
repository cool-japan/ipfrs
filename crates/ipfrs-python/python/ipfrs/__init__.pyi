"""
IPFRS Python Bindings

Inter-Planetary File Rust System - Content-addressed storage with
semantic search and logic programming capabilities.
"""

from typing import Optional, List, Tuple, Dict

class NodeConfig:
    """Configuration for IPFRS node"""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        enable_network: bool = False,
        enable_semantic: bool = True,
        enable_tensorlogic: bool = True
    ) -> None:
        """
        Create a new node configuration

        Args:
            storage_path: Path to storage directory (optional)
            enable_network: Enable networking features (default: False)
            enable_semantic: Enable semantic search (default: True)
            enable_tensorlogic: Enable logic programming (default: True)
        """
        ...

    @staticmethod
    def default() -> "NodeConfig":
        """Create default configuration"""
        ...

class Node:
    """IPFRS Node - main interface for all operations"""

    def __init__(self, config: Optional[NodeConfig] = None) -> None:
        """
        Create a new IPFRS node

        Args:
            config: Node configuration (optional, uses default if not provided)
        """
        ...

    def start(self) -> None:
        """Start the node (required before use)"""
        ...

    def stop(self) -> None:
        """Stop the node"""
        ...

    def put_block(self, data: bytes) -> Cid:
        """
        Add a block to storage

        Args:
            data: Bytes to store

        Returns:
            Content identifier of the stored block
        """
        ...

    def get_block(self, cid: Cid) -> Optional[Block]:
        """
        Get a block from storage

        Args:
            cid: Content identifier to retrieve

        Returns:
            The block if found, None otherwise
        """
        ...

    def has_block(self, cid: Cid) -> bool:
        """
        Check if a block exists

        Args:
            cid: Content identifier to check

        Returns:
            True if block exists, False otherwise
        """
        ...

    def delete_block(self, cid: Cid) -> None:
        """
        Delete a block from storage

        Args:
            cid: Content identifier to delete
        """
        ...

    def index_content(self, cid: Cid, embedding: List[float]) -> int:
        """
        Index content for semantic search

        Args:
            cid: Content identifier
            embedding: Vector embedding (list of floats)

        Returns:
            Index of the added vector
        """
        ...

    def search_similar(self, query: List[float], k: int) -> List[Tuple[Cid, float]]:
        """
        Search for similar content

        Args:
            query: Query vector embedding (list of floats)
            k: Number of results to return

        Returns:
            List of tuples (cid, score)
        """
        ...

    def search_filtered(
        self,
        query: List[float],
        k: int,
        filter: Optional[Filter] = None
    ) -> List[Tuple[Cid, float]]:
        """
        Search with filters

        Args:
            query: Query vector embedding (list of floats)
            k: Number of results to return
            filter: Search filter (optional)

        Returns:
            List of tuples (cid, score)
        """
        ...

    def add_fact(self, fact: Predicate) -> None:
        """
        Add a fact to the knowledge base

        Args:
            fact: Predicate representing the fact
        """
        ...

    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the knowledge base

        Args:
            rule: Rule to add
        """
        ...

    def infer(self, goal: Predicate) -> List[Substitution]:
        """
        Run inference query

        Args:
            goal: Predicate to prove

        Returns:
            List of substitutions that satisfy the goal
        """
        ...

    def prove(self, goal: Predicate) -> Optional[Proof]:
        """
        Generate a proof for a goal

        Args:
            goal: Predicate to prove

        Returns:
            Proof if goal can be proven, None otherwise
        """
        ...

    def verify_proof(self, proof: Proof) -> bool:
        """
        Verify a proof

        Args:
            proof: Proof to verify

        Returns:
            True if proof is valid, False otherwise
        """
        ...

    def kb_stats(self) -> Dict[str, int]:
        """
        Get knowledge base statistics

        Returns:
            Statistics including number of facts and rules
        """
        ...

    def save_semantic_index(self, path: str) -> None:
        """
        Save semantic index to disk

        Args:
            path: Path to save index file
        """
        ...

    def load_semantic_index(self, path: str) -> None:
        """
        Load semantic index from disk

        Args:
            path: Path to index file
        """
        ...

    def save_kb(self, path: str) -> None:
        """
        Save knowledge base to disk

        Args:
            path: Path to save KB file
        """
        ...

    def load_kb(self, path: str) -> None:
        """
        Load knowledge base from disk

        Args:
            path: Path to KB file
        """
        ...

class Block:
    """Content-addressed block"""

    def __init__(self, data: bytes) -> None:
        """
        Create a new block from data

        Args:
            data: Bytes to store
        """
        ...

    def data(self) -> bytes:
        """
        Get block data

        Returns:
            Block data
        """
        ...

    def cid(self) -> Cid:
        """
        Get block CID

        Returns:
            Content identifier
        """
        ...

    def size(self) -> int:
        """
        Get block size

        Returns:
            Size in bytes
        """
        ...

class Cid:
    """Content Identifier"""

    @staticmethod
    def parse(s: str) -> "Cid":
        """
        Parse CID from string

        Args:
            s: CID string representation

        Returns:
            Parsed CID
        """
        ...

    def __str__(self) -> str:
        """Convert CID to string"""
        ...

    def __repr__(self) -> str:
        """Convert CID to repr"""
        ...

class Term:
    """Logical term"""

    @staticmethod
    def int(value: int) -> "Term":
        """
        Create a constant integer term

        Args:
            value: Integer value

        Returns:
            Integer constant
        """
        ...

    @staticmethod
    def float(value: float) -> "Term":
        """
        Create a constant float term

        Args:
            value: Float value

        Returns:
            Float constant
        """
        ...

    @staticmethod
    def string(value: str) -> "Term":
        """
        Create a constant string term

        Args:
            value: String value

        Returns:
            String constant
        """
        ...

    @staticmethod
    def var(name: str) -> "Term":
        """
        Create a variable term

        Args:
            name: Variable name

        Returns:
            Variable
        """
        ...

    def __str__(self) -> str:
        """String representation"""
        ...

class Predicate:
    """Logical predicate"""

    def __init__(self, name: str, args: List[Term]) -> None:
        """
        Create a new predicate

        Args:
            name: Predicate name
            args: List of terms
        """
        ...

    def __str__(self) -> str:
        """String representation"""
        ...

class Rule:
    """Logical rule"""

    @staticmethod
    def fact(head: Predicate) -> "Rule":
        """
        Create a fact (rule with no body)

        Args:
            head: Head predicate

        Returns:
            Fact rule
        """
        ...

    @staticmethod
    def rule(head: Predicate, body: List[Predicate]) -> "Rule":
        """
        Create a rule with body

        Args:
            head: Head predicate
            body: List of body predicates

        Returns:
            New rule
        """
        ...

    def __str__(self) -> str:
        """String representation"""
        ...

class Proof:
    """Proof tree"""

    def __str__(self) -> str:
        """String representation"""
        ...

class Substitution:
    """Variable substitution"""

    def bindings(self) -> Dict[str, str]:
        """
        Get bindings as dictionary

        Returns:
            Variable name to term mappings
        """
        ...

    def __str__(self) -> str:
        """String representation"""
        ...

class Filter:
    """Search filter"""

    @staticmethod
    def min_score(min_score: float) -> "Filter":
        """
        Create a filter with minimum score threshold

        Args:
            min_score: Minimum similarity score

        Returns:
            New filter
        """
        ...

    @staticmethod
    def max_score(max_score: float) -> "Filter":
        """
        Create a filter with maximum score threshold

        Args:
            max_score: Maximum similarity score

        Returns:
            New filter
        """
        ...

__all__ = [
    "Node",
    "NodeConfig",
    "Block",
    "Cid",
    "Term",
    "Predicate",
    "Rule",
    "Proof",
    "Substitution",
    "Filter",
]
