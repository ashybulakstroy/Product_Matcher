"""Local invoice chain: contract and tools that start AFTER a validated receipt.

The modules in this package read an already verified Torgsoft receipt snapshot
and produce structured data for 1C nomenclature matching. They never re-run
OCR, never mutate the receipt file, and never write real documents into 1C.
"""
