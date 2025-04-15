#!/usr/bin/env python3
"""
Memory Log Analysis Command Line Interface

This script provides a unified command-line interface for parsing and analyzing
memory allocation logs from legate-sparse.

Optional Dependencies:
- pandas, matplotlib, seaborn: Required for visualizations
- openpyxl: Required for Excel export
"""

import argparse
import os
import sys

from memlog_analysis import export_to_csv, export_to_excel, visualize_allocations
from memlog_parser import (
    filter_allocations,
    parse_memlog,
    print_description_group,
    print_size_group,
    update_type_sizes,
)


def check_dependencies(format: str) -> bool:
    """
    Check if required dependencies are available for the requested format.

    Args:
        format: Requested output format

    Returns:
        bool: True if all required dependencies are available
    """
    if format == "excel":
        try:
            import openpyxl  # noqa:  F401
        except ImportError:
            print(
                "Error: Excel export requires openpyxl. Please install it with: pip install openpyxl"
            )
            return False

    if format == "visualization":
        try:
            import matplotlib  # noqa:  F401
            import pandas  # noqa:  F401
            import seaborn  # noqa:  F401
        except ImportError:
            print("Error: Visualization requires pandas, matplotlib, and seaborn.")
            print("Please install them with: pip install pandas matplotlib seaborn")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Parse and analyze memory allocation logs from legate-sparse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage (print to screen):
    python memlog_cli.py memlog.txt

  Group by size:
    python memlog_cli.py memlog.txt --group-by size

  Ignore specific descriptions:
    python memlog_cli.py memlog.txt --ignore-descriptions "ThrustAllocator::allocate" "buffer1"

  Set minimum memory size:
    python memlog_cli.py memlog.txt --min-mb 1.0

  Show only unique memory sizes within each group:
    python memlog_cli.py memlog.txt --unique-mb-only

  Custom similarity threshold:
    python memlog_cli.py memlog.txt --unique-mb-only --similarity-threshold 5.0

  Configure type sizes:
    python memlog_cli.py memlog.txt --index-ty-size 4 --val-ty-size 8

  Export to CSV (in addition to printing):
    python memlog_cli.py memlog.txt --format csv --output-dir ./analysis

  Export to Excel (requires openpyxl):
    python memlog_cli.py memlog.txt --format excel --output-dir ./analysis

  Create visualizations (requires pandas, matplotlib, seaborn):
    python memlog_cli.py memlog.txt --format visualization --output-dir ./analysis

  Combine multiple options:
    python memlog_cli.py memlog.txt --unique-mb-only --similarity-threshold 1.0 --min-mb 1.0 --ignore-descriptions "ThrustAllocator::allocate" --format csv
""",
    )

    # Required arguments
    parser.add_argument("file", help="Path to the memory log file")

    # Filtering options
    parser.add_argument(
        "--ignore-descriptions",
        nargs="+",
        default=[],
        help="List of descriptions to ignore",
    )
    parser.add_argument(
        "--min-mb",
        type=float,
        default=0.0,
        help="Minimum memory size in MB to include (default: 0.0)",
    )
    parser.add_argument(
        "--unique-mb-only",
        action="store_true",
        help="Only show unique memory sizes in MB",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=5.0,
        help="Percentage threshold for considering memory sizes similar (default: 5.0%%)",
    )

    # Type size configuration
    parser.add_argument(
        "--index-ty-size",
        type=int,
        default=8,
        help="Size of INDEX_TY in bytes (default: 8)",
    )
    parser.add_argument(
        "--val-ty-size",
        type=int,
        default=8,
        help="Size of VAL_TY in bytes (default: 8)",
    )

    # Output options
    parser.add_argument(
        "--group-by",
        choices=["description", "size"],
        default="description",
        help="Group allocations by description or size (default: description)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "excel", "visualization"],
        help="Additional output format (optional)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output files (default: current directory)",
    )

    args = parser.parse_args()

    try:
        # Update type sizes based on command line arguments
        update_type_sizes(args.index_ty_size, args.val_ty_size)

        # Parse the log file
        allocations = parse_memlog(args.file)

        # Filter allocations based on criteria
        filtered_allocations = filter_allocations(
            allocations,
            ignore_descriptions=set(args.ignore_descriptions),
            min_mb=args.min_mb,
        )

        # Print results to screen
        print("\nMemory Allocation Analysis:")
        print("=" * 50)
        print(f"Using INDEX_TY size: {args.index_ty_size} bytes")
        print(f"Using VAL_TY size: {args.val_ty_size} bytes")
        print("=" * 50)

        if args.group_by == "description":
            print_description_group(
                filtered_allocations,
                unique_mb_only=args.unique_mb_only,
                threshold_percent=args.similarity_threshold,
            )
        else:
            print_size_group(
                filtered_allocations,
                unique_mb_only=args.unique_mb_only,
                threshold_percent=args.similarity_threshold,
            )

        # Handle additional output formats if requested
        if args.format:
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)

            # Check dependencies if needed
            if args.format in ["excel", "visualization"]:
                if not check_dependencies(args.format):
                    return 1

            # Generate requested output
            success = True
            if args.format == "csv":
                export_to_csv(
                    filtered_allocations,
                    f"{args.output_dir}/memory_analysis.csv",
                    group_by=args.group_by,
                    unique_mb_only=args.unique_mb_only,
                    threshold_percent=args.similarity_threshold,
                )
            elif args.format == "excel":
                if not export_to_excel(
                    filtered_allocations,
                    f"{args.output_dir}/memory_analysis.xlsx",
                ):
                    success = False
            elif args.format == "visualization":
                if not visualize_allocations(
                    filtered_allocations,
                    args.output_dir,
                    unique_mb_only=args.unique_mb_only,
                    threshold_percent=args.similarity_threshold,
                ):
                    success = False

            return 0 if success else 1

        return 0

    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
