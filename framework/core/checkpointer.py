from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

import config


def get_checkpointer() -> BaseCheckpointSaver:
    if config.APP_ENV == "prod":
        if not config.POSTGRES_URL:
            raise ValueError("POSTGRES_URL must be set when APP_ENV=prod")
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        return AsyncPostgresSaver.from_conn_string(config.POSTGRES_URL)
    return MemorySaver()
