from datetime import datetime
from typing import Optional, List

from app import db
from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model, UserMixin):
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True)
    password_hash: Mapped[str]
    fullname: Mapped[str]
    nickname: Mapped[Optional[str]] = mapped_column(String(80))
    lifetime_score: Mapped[int] = mapped_column(insert_default=0)

    scores: Mapped[List["QuizScore"]] = relationship(back_populates="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

class Subject(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str]

    quizzes: Mapped[List["Quiz"]] = relationship(back_populates="subject", lazy=True)

class Quiz(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str]
    date_of_quiz: Mapped[datetime]
    duration: Mapped[int]

    subject_id = mapped_column(ForeignKey("subject.id"))
    subject: Mapped["Subject"] = relationship(back_populates="quizzes")
    questions: Mapped[List["QuizQuestion"]] = relationship(back_populates="quiz")
    scores: Mapped[List["QuizScore"]] = relationship(back_populates="quiz", lazy=True)

class QuizQuestion(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    question_statement: Mapped[str]

    quiz_id = mapped_column(ForeignKey("quiz.id"))
    quiz: Mapped["Quiz"] = relationship(back_populates="questions")
    choices: Mapped[List["QuizChoice"]] = relationship(back_populates="question")

class QuizChoice(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str]
    is_correct: Mapped[bool] = mapped_column(insert_default=False)

    quiz_id = mapped_column(ForeignKey("quiz_question.id"))
    question: Mapped["QuizQuestion"] = relationship(back_populates="choices")

class QuizScore(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    total_scored: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(insert_default=datetime.now())
    updated_at: Mapped[datetime] = mapped_column(insert_default=datetime.now())

    quiz_id = mapped_column(ForeignKey("quiz.id"))
    quiz: Mapped["Quiz"] = relationship(back_populates="scores")
    user_id = mapped_column(ForeignKey("user.id"))
    user: Mapped["User"] = relationship(back_populates="scores")
