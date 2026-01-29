from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse
from app.schemas import *
import argilla as ag
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARGILLA_API_URL, ARGILLA_API_KEY
import tempfile
import os
import shutil
import ray
from ray import job_submission

# Initialize Argilla client
try:
    client = ag.Argilla(api_url=ARGILLA_API_URL, api_key=ARGILLA_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Argilla: {str(e)}")

app = FastAPI(title="Argilla Management API")


# Pydantic models for request/response


# Routes for user management

async def cleanup(path: str):
    shutil.rmtree(path)


@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    """
    Create a new user in Argilla.

    Args:
        user: User creation details

    Returns:
        Created user information
    """
    try:
        # Create user via Argilla API
        new_user = ag.User(
            username=user.username,
            password=user.password,
            first_name=user.first_name,
            last_name=user.last_name,
            client=client,
        )
        created_user = new_user.create()
        return UserResponse(
            id=str(created_user.id),
            username=new_user.username,
            first_name=new_user.first_name,
            last_name=new_user.last_name,
            message="User created successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create user: {str(e)}",
        )


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """
    Retrieve user information by ID.

    Args:
        user_id: The user ID

    Returns:
        User information
    """
    try:
        user = client.users(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )
        return UserResponse(
            id=str(user.id),
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            message="Successfully fectched user details",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user: {str(e)}",
        )


@app.put("/users/{user_id}", response_model=UserResponse)  # TO DO
async def update_user(user_id: str, user_update: UserUpdate):
    """
    Update user information.

    Args:
        user_id: The user ID
        user_update: Fields to update

    Returns:
        Updated user information
    """
    try:
        user = client.users.get_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        # Update only provided fields
        update_data = user_update.dict(exclude_unset=True)
        updated_user = client.users.update(user_id, **update_data)

        return UserResponse(
            id=str(updated_user.id),
            username=updated_user.username,
            email=updated_user.email,
            first_name=updated_user.first_name,
            last_name=updated_user.last_name,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update user: {str(e)}",
        )


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    """
    Delete a user by ID.

    Args:
        user_id: The user ID to delete
    """
    try:
        user = client.users.get_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        client.users.delete(user_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to delete user: {str(e)}",
        )


@app.post("/workspace/{workspace_name}")
async def create_workspace(workspace_name: str):
    try:
        workspace_to_create = ag.Workspace(name=workspace_name, client=client)
        created_workspace = workspace_to_create.create()
        return WorkspaceResponse(id=created_workspace.id, name=created_workspace.name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create workspace: {str(e)}",
        )


@app.delete("/workspace/{workspace_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(workspace_name: str):
    try:
        workspace_to_delete = client.workspaces(name=workspace_name)
        deleted_workspace = workspace_to_delete.delete()
        return {"delete_workspace": delete_workspace}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create workspace: {str(e)}",
        )


@app.post("/users/assign_ws")
async def assign_user_to_workspace(assign_ws: UserWorkspace):
    try:
        ws = client.workspaces(assign_ws.workspace_name)
        added_user = ws.add_user(assign_ws.user_name)
        return UserResponse(
            id=str(added_user.id),
            username=added_user.username,
            message=f"Successfully added to workspace {assign_ws.workspace_name}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add user to workspace: {str(e)}",
        )


@app.delete("/users/remove_ws")
async def remove_user_from_workspace(remove_ws: UserWorkspace):
    try:
        ws = client.workspaces(remove_ws.workspace_name)
        removed_user = ws.remove(remove_ws.user_name)
        return UserResponse(
            id=removed_user.id,
            username=removed_user.username,
            message=f"Successfully added to workspace {remove_ws.workspace_name}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add user to workspace: {str(e)}",
        )


@app.post("/dataset/create")
async def create_dataset(ds_ws: CreateDataset):
    try:
        if ds_ws.dataset_type == "chat":
            settings = ag.Settings(
                guidelines="Analyse the question and check if soultion is valid",
                fields=[ag.ChatField(
                        name="chat",
                        title="Chat",
                        use_markdown=True,
                        required=True,
                        description="Field description",
                    )],
                questions=[ag.LabelQuestion(
                            name="is_response_correct",
                            title="Is the response correct?",
                            labels=["yes", "no"],
                        ),
                        ag.LabelQuestion(
                            name="out_of_guardrails",
                            title="Did the model answered something out of the ordinary?",
                            description="If the model answered something unrelated to Argilla SDK",
                            labels=["yes", "no"],
                        ),
                        ag.TextQuestion(
                            name="feedback",
                            title="Let any feedback here",
                            description="This field should be used to report any feedback that can be useful",
                            required=False
                        ),]
            )
        dataset = ag.Dataset(name=ds_ws.dataset_name,
                            workspace=ds_ws.workspace_name,
                            settings=settings)
        dataset.create()
        return {"message": "successfully created dataset"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add dataset to workspace: {str(e)}",
        )

@app.post("/record/chat/create") #TODO: add return response and complete error handling
async def create_chat_record(record: CreateChatRecord):
    dataset = client.datasets(name = record.dataset_name,workspace= record.workspace_name)
    record = [ag.Record(
        fields ={
            "chat" :[
                {"role": "developer", "content": record.question},
                {"role": "assistant", "content": record.answer}
            ]
        },
        metadata=[ag.TermsMetadataProperty(name=key,title=value) for key, value in record.items()]
    )]
    dataset.records.log(record)


@app.post("/export/dataset")#Handled removing completed datasets while exporting.
async def export_data_set(export_ds: ExportDataset,background_tasks: BackgroundTasks):
    dataset = client.datasets(name = export_ds.dataset_name,workspace= export_ds.workspace_name)
    if export_ds.export_type == "completed":
        query = ag.Query(
            filter=ag.Filter(
                ("status", "==", "completed")
            )
        )
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False,dir=".")
        temp_file_path = temp_file.name
        temp_file.close()

        query_filtered_record = dataset.records(query=query).to_datasets()
        query_filtered_record.to_json(path_or_buf=temp_file_path)
        background_tasks.add_task(os.remove, temp_file_path)
        return FileResponse(path=temp_file_path, filename="records.json", media_type="application/json")
        
    if export_ds.export_type == "full":   
        temp_dir = tempfile.mkdtemp(dir=".")
        dataset_tempdir = tempfile.mkdtemp(dir=".")
        try:
            dataset.to_disk(path=temp_dir)
            # Create zip outside temp_dir to avoid nesting
            zip_path = os.path.abspath(os.path.join(dataset_tempdir, f"{export_ds.dataset_name}.zip"))
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_dir)
            background_tasks.add_task(cleanup, dataset_tempdir)
            return FileResponse(path=zip_path, filename=f"{export_ds.dataset_name}.zip", media_type="application/zip")
        finally:
            shutil.rmtree(temp_dir)
        
@app.post("/train/model")
async def train_model(train_request: TrainModelRequest):
    """
    Submit a classification model training job to Ray cluster.
    
    Args:
        train_request: Training request with model_name and dataset_name
        
    Returns:
        Job submission details
    """
    try:
        
        # Initialize Ray job submission client
        ray_client = job_submission.JobSubmissionClient("http://127.0.0.1:8265")
        
        # Submit training job to Ray cluster
        job_id = ray_client.submit_job(
            entrypoint=f"python train_classification_model.py --model_name {train_request.model_name} --dataset_name {train_request.dataset_name} --workspace_name {train_request.workspace_name}",
            runtime_env={
                "pip": ["argilla", "transformers","mlflow","datasets","requests","torch","python-dotenv"],
                "working_dir": "./app/resource",
            }
        )
        
        return {
            "message": "Training job submitted successfully",
            "job_id": job_id,
            "model_name": train_request.model_name,
            "dataset_name": train_request.dataset_name
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit training job: {str(e)}"
        )
    

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "argilla_url": ARGILLA_API_URL}
