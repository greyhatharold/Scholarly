import sqlite3
from typing import Optional
import numpy as np
from contextlib import contextmanager
from src.config.config import DatabaseConfig

class DatabaseConnection:
    """Handles SQLite database connections with vector storage optimization"""
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_path = config.db_path
        self._connection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Creates necessary tables and indexes"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_data BLOB NOT NULL,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON vector_store(created_at)")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        try:
            yield self._connection
        finally:
            self._connection.close()
            self._connection = None 