"""Connection helper for the local user-space MobilityDB/PostgreSQL instance."""
import psycopg


def get_connection(dsn: str) -> psycopg.Connection:
    """Open an autocommit connection to the local MobilityDB/PostgreSQL instance.

    Args:
        dsn: PostgreSQL connection string, e.g.
            ``postgresql://postgres@localhost:5434/mobilitydb``.

    Returns:
        An open, autocommit psycopg connection.
    """
    return psycopg.connect(dsn, autocommit=True)
