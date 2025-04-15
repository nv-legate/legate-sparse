#!/usr/bin/env python3
"""
Memory Log Parser Core Module

This module contains the core functionality for parsing memory allocation logs
from legate-sparse, including data structures and basic parsing functions.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

# Dictionary mapping data types to their sizes in bytes
TYPE_SIZES = {
    "char": 1,
    "int32_t": 4,
    "int64_t": 8,
    "float": 4,
    "double": 8,
    "float32": 4,
    "float64": 8,
    "INDEX_TY": 8,  # Default to int64_t
    "VAL_TY": 8,  # Default to double
}


def update_type_sizes(index_ty_size: int = 8, val_ty_size: int = 8) -> None:
    """
    Update the sizes of INDEX_TY and VAL_TY in the TYPE_SIZES dictionary.

    Args:
        index_ty_size: Size of INDEX_TY in bytes (default: 8)
        val_ty_size: Size of VAL_TY in bytes (default: 8)
    """
    TYPE_SIZES["INDEX_TY"] = index_ty_size
    TYPE_SIZES["VAL_TY"] = val_ty_size


@dataclass
class BufferAllocation:
    """
    Represents a single buffer allocation from the memory log.

    Attributes:
        timestamp: Time of allocation in seconds
        file: Source file where allocation occurred
        line: Line number in the source file
        size: Number of elements allocated
        type: Data type of the elements
        description: Description of the allocation
    """

    timestamp: float
    file: str
    line: int
    size: int
    type: str
    description: str

    def total_bytes(self) -> int:
        """Calculate total bytes allocated including data type size."""
        type_size = TYPE_SIZES.get(self.type, 1)  # Default to 1 byte if type not found
        return self.size * type_size

    def total_mb(self) -> float:
        """Calculate total memory in MB."""
        return self.total_bytes() / (1024 * 1024)


def are_similar_sizes(size1: float, size2: float, threshold_percent: float) -> bool:
    """
    Check if two sizes are similar within the given percentage threshold.

    Args:
        size1: First size in MB
        size2: Second size in MB
        threshold_percent: Maximum allowed percentage difference

    Returns:
        True if sizes are within threshold, False otherwise
    """
    if size1 == 0 or size2 == 0:
        return size1 == size2
    percent_diff = abs(size1 - size2) / min(size1, size2) * 100
    return percent_diff <= threshold_percent


def parse_memlog(file_path: str) -> List[BufferAllocation]:
    """
    Parse the memory log file and extract buffer allocation information.

    Args:
        file_path: Path to the memory log file

    Returns:
        List of BufferAllocation objects

    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    pattern = r"\[.*?\]\s+(\d+\.\d+)\s+\{3\}\{legate-sparse\}:\s+Buffer allocation at ([^:]+):(\d+)\s+-\s+Size:\s+(\d+)\s+Type:\s+([^\s]+)\s+Description:\s+(.+)"

    allocations = []
    seen_entries = set()  # To track unique entries

    with open(file_path, "r") as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                # Create a unique key for the entry
                entry_key = (
                    match.group(2),  # file
                    match.group(3),  # line
                    match.group(4),  # size
                    match.group(5),  # type
                    match.group(6),  # description
                )

                # Only add if we haven't seen this exact entry before
                if entry_key not in seen_entries:
                    seen_entries.add(entry_key)
                    allocation = BufferAllocation(
                        timestamp=float(match.group(1)),
                        file=match.group(2),
                        line=int(match.group(3)),
                        size=int(match.group(4)),
                        type=match.group(5),
                        description=match.group(6),
                    )
                    allocations.append(allocation)

    return allocations


def group_by_description(
    allocations: List[BufferAllocation],
) -> Dict[str, List[BufferAllocation]]:
    """
    Group allocations by their description.

    Args:
        allocations: List of BufferAllocation objects

    Returns:
        Dictionary mapping descriptions to lists of allocations
    """
    grouped = defaultdict(list)
    for alloc in allocations:
        grouped[alloc.description].append(alloc)
    return dict(grouped)


def group_by_size(
    allocations: List[BufferAllocation],
) -> Dict[int, List[BufferAllocation]]:
    """
    Group allocations by their size in elements.

    Args:
        allocations: List of BufferAllocation objects

    Returns:
        Dictionary mapping sizes to lists of allocations
    """
    grouped = defaultdict(list)
    for alloc in allocations:
        grouped[alloc.size].append(alloc)
    return dict(grouped)


def filter_allocations(
    allocations: List[BufferAllocation],
    ignore_descriptions: Set[str] = None,
    min_mb: float = 0.0,
) -> List[BufferAllocation]:
    """
    Filter allocations based on description and minimum size criteria.

    Args:
        allocations: List of BufferAllocation objects
        ignore_descriptions: Set of descriptions to ignore
        min_mb: Minimum memory size in MB to include

    Returns:
        Filtered list of BufferAllocation objects
    """
    if ignore_descriptions is None:
        ignore_descriptions = set()

    filtered = []
    for alloc in allocations:
        if alloc.description not in ignore_descriptions and alloc.total_mb() >= min_mb:
            filtered.append(alloc)
    return filtered


def print_description_group(
    allocations: List[BufferAllocation],
    unique_mb_only: bool = False,
    threshold_percent: float = 5.0,
):
    """
    Print allocations grouped by description.

    Args:
        allocations: List of BufferAllocation objects
        unique_mb_only: If True, only show unique memory sizes
        threshold_percent: Percentage threshold for considering sizes similar
    """
    grouped_by_desc = group_by_description(allocations)

    for desc, allocs in grouped_by_desc.items():
        desc_total_bytes = sum(alloc.total_bytes() for alloc in allocs)
        max_bytes = max(alloc.total_bytes() for alloc in allocs)
        print(f"\n{desc}:")
        print(
            f"  Total bytes (includes non-unique allocs): {desc_total_bytes / (1024*1024):.2f} MB"
        )
        print(f"  Max bytes  : {max_bytes / (1024*1024):.2f} MB")

        # Track seen entries for this description
        seen_entries = set()
        seen_mb_sizes = set()

        for alloc in allocs:
            mb_size = alloc.total_mb()

            # If unique_mb_only is enabled, check for similar memory sizes
            if unique_mb_only:
                is_similar = any(
                    are_similar_sizes(mb_size, seen_size, threshold_percent)
                    for seen_size in seen_mb_sizes
                )
                if is_similar:
                    continue
                seen_mb_sizes.add(mb_size)

            # Create a unique key for this entry
            entry_key = (alloc.size, alloc.type, alloc.file, alloc.line)

            # Skip if we've seen this exact entry before
            if entry_key in seen_entries:
                continue
            seen_entries.add(entry_key)

            print(
                f"  - Size: {alloc.size} elements, Type: {alloc.type} ({TYPE_SIZES.get(alloc.type, 1)} bytes), "
                f"Total: {mb_size:.2f} MB, "
                f"File: {alloc.file}:{alloc.line}, Time: {alloc.timestamp}"
            )


def print_size_group(
    allocations: List[BufferAllocation],
    unique_mb_only: bool = False,
    threshold_percent: float = 2.0,
):
    """
    Print allocations grouped by size.

    Args:
        allocations: List of BufferAllocation objects
        unique_mb_only: If True, only show unique memory sizes
        threshold_percent: Percentage threshold for considering sizes similar
    """
    grouped_by_size_dict = group_by_size(allocations)
    seen_mb_sizes = set()

    for size, allocs in sorted(
        grouped_by_size_dict.items(), key=lambda x: x[0], reverse=True
    ):
        size_total_bytes = sum(alloc.total_bytes() for alloc in allocs)
        max_bytes = max(alloc.total_bytes() for alloc in allocs)

        print(f"\nSize: {size} elements:")
        print(
            f"  Total bytes (includes non-unique allocs): {size_total_bytes / (1024*1024):.2f} MB"
        )
        print(f"  Max bytes  : {max_bytes / (1024*1024):.2f} MB")

        for alloc in allocs:
            mb_size = alloc.total_mb()
            if unique_mb_only:
                # Check if this size is similar to any previously seen size
                is_similar = any(
                    are_similar_sizes(mb_size, seen_size, threshold_percent)
                    for seen_size in seen_mb_sizes
                )
                if is_similar:
                    continue
            seen_mb_sizes.add(mb_size)
            print(
                f"  - Type: {alloc.type} ({TYPE_SIZES.get(alloc.type, 1)} bytes), "
                f"Description: {alloc.description}, "
                f"File: {alloc.file}:{alloc.line}"
            )
