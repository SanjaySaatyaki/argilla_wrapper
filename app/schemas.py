from pydantic import BaseModel, Field
from uuid import UUID, uuid4

class UserCreate(BaseModel):
    username: str
    password: str
    first_name: str | None = None
    last_name: str | None = None


class UserUpdate(BaseModel):
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    password: str | None = None


class UserResponse(BaseModel):
    id: str
    username: str
    first_name: str | None = None
    last_name: str | None = None
    message: str | None = None
    class Config:
        from_attributes = True


class WorkspaceResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str

class UserWorkspace(BaseModel):
    user_name: str
    workspace_name: str

class CreateDataset(BaseModel):
    workspace_name: str
    dataset_name: str
    dataset_type: str


class CreateChatRecord(BaseModel):
    question: str
    answer: str
    meta_data: dict
    dataset_name: str
    workspace_name: str