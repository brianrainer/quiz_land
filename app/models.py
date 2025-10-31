from datetime import datetime
from typing import Optional

from app import db
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

class User(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True, nullable=False)
    password: Mapped[str] = mapped_column(nullable=False)
    fullname: Mapped[str]

    created_at: Mapped[datetime] = mapped_column(insert_default=datetime.now())
