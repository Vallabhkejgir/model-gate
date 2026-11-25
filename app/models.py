"""Pydantic schemas for OpenAI-compatible endpoints and Admin Control Plane."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A single message in an OpenAI chat completion request."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, List[Dict[str, Any]], None] = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request payload."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage] = Field(..., min_length=1)
    stream: bool = False
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    n: Optional[int] = 1
    user: Optional[str] = None


class CreateVirtualKeyRequest(BaseModel):
    """Admin request to issue a new virtual consumer API key."""

    name: str = Field(..., min_length=1, max_length=120)
    allowed_models: Optional[List[str]] = None
    forced_system_prompt: Optional[str] = None
    max_tokens_per_request: int = Field(default=1024, ge=1, le=65536)
    max_input_chars: int = Field(default=16000, ge=100, le=2000000)
    rpm_limit: int = Field(default=15, ge=1, le=10000)
    token_budget: Optional[int] = Field(default=None, ge=0)
    expires_at: Optional[str] = None


class UpdateVirtualKeyRequest(BaseModel):
    """Admin request to update policy or reset counters on an existing consumer key."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    allowed_models: Optional[List[str]] = None
    forced_system_prompt: Optional[str] = None
    max_tokens_per_request: Optional[int] = Field(default=None, ge=1, le=65536)
    max_input_chars: Optional[int] = Field(default=None, ge=100, le=2000000)
    rpm_limit: Optional[int] = Field(default=None, ge=1, le=10000)
    token_budget: Optional[int] = Field(default=None, ge=0)
    is_active: Optional[bool] = None
    expires_at: Optional[str] = None
    reset_usage: Optional[bool] = False
