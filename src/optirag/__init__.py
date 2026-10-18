__version__ = "0.1.0"


def main() -> None:
    from .ingest import main as ingest_main

    ingest_main()
