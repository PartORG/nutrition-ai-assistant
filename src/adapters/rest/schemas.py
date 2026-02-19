"""Pydantic models for REST API request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Auth ---

class RegisterBody(BaseModel):
    login: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    name: str = ""
    surname: str = ""
    age: int = 0
    gender: str = ""
    caretaker: str = ""
    health_condition: str = ""


class LoginBody(BaseModel):
    login: str
    password: str


class RefreshBody(BaseModel):
    token: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    role: str


# --- Recommendations ---

class RecommendationBody(BaseModel):
    query: str = Field(..., min_length=1)


class RecommendationOut(BaseModel):
    summary: str
    raw_recommendations: str


# --- Conversations ---

class ConversationOut(BaseModel):
    conversation_id: str
    title: str
    last_message_at: str
    created_at: str


class MessageOut(BaseModel):
    id: int | None
    role: str
    content: str
    created_at: str


# --- Profile ---

class UserOut(BaseModel):
    name: str
    surname: str
    user_name: str
    age: int
    gender: str
    caretaker: str


class ProfileOut(BaseModel):
    preferences: str
    health_condition: str
    restrictions: str
    created_at: str


class MedicalAdviceOut(BaseModel):
    health_condition: str
    medical_advice: str
    dietary_limit: str
    avoid: str


# --- Image ---

class ImageAnalysisOut(BaseModel):
    detected_ingredients: list[str]
    recommendation_summary: str | None = None
