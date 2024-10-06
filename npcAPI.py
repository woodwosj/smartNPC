from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_function
import uuid
import os
import openai

#Extract OpenAI API key from local system environmental variable.
os.getenv('OPENAI_API_KEY')

# Initialize FastAPI app
app = FastAPI()
# Initialize Chroma vector database
client = chromadb.Client()
db = client.get_or_create_collection(name="npc_memory")

# NPC Model
class NPC(BaseModel):
    name: str
    species: str
    age: int
    gender: str
    skills: List[str]
    background: str
    abilities: List[str]
    true_name: str

# Action Model
class ActionRequest(BaseModel):
    actions: List[str]
    context: Optional[str] = None

# In-memory storage for NPC profiles
npc_storage = {}

# Helper function to create embeddings for memory
embedding_fn = embedding_function()

# Create NPC endpoint
@app.post("/npc/create")
async def create_npc(npc: NPC):
    npc_id = str(uuid.uuid4())
    npc_storage[npc_id] = npc
    return {"npc_id": npc_id, "message": "NPC created successfully"}

# Get NPC endpoint
@app.get("/npc/{npc_id}")
async def get_npc(npc_id: str):
    npc = npc_storage.get(npc_id)
    if npc is None:
        raise HTTPException(status_code=404, detail="NPC not found")
    return npc

# Update NPC memory endpoint
@app.post("/npc/{npc_id}/action")
async def perform_action(npc_id: str, action_request: ActionRequest):
    npc = npc_storage.get(npc_id)
    if npc is None:
        raise HTTPException(status_code=404, detail="NPC not found")

    # Retrieve relevant memories for context
    memories = db.query(where={"npc_id": npc_id})
    memory_context = "\n".join([memory["response"] for memory in memories])

    # Generate response for each action based on context
    responses = []
    for action in action_request.actions:
        if action == "speak":
            response = generate_dialogue(npc, action_request.context, memory_context)
        elif action == "emote":
            response = generate_emotion(npc, action_request.context, memory_context)
        else:
            response = {"error": "Unknown action"}
        responses.append(response)

    # Store action in the vector database for memory
    memory_id = str(uuid.uuid4())
    embedding = embedding_fn(action_request.context) if action_request.context else embedding_fn(str(responses))
    db.add(ids=[memory_id], embeddings=[embedding], metadatas=[{"npc_id": npc_id, "actions": action_request.actions, "responses": responses}])

    return {"npc_id": npc_id, "responses": responses}

# Function to generate dialogue
def generate_dialogue(npc: NPC, context: Optional[str], memory_context: str) -> dict:
    # Use GPT-4o-mini to generate dialogue with intelligent token usage
    max_input_tokens = 128000 - 16384  # Reserve 16384 tokens for output
    character_card = f"Character Card:\nName: {npc.name}\nSpecies: {npc.species}\nAge: {npc.age}\nGender: {npc.gender}\nSkills: {', '.join(npc.skills)}\nBackground: {npc.background}\nAbilities: {', '.join(npc.abilities)}\n"
    combined_context = f"Memories:\n{memory_context}\n\nContext: {context}\n" if context else f"Memories:\n{memory_context}\n"
    available_tokens = max_input_tokens - len(character_card.split()) - len(combined_context.split())
    prompt = f"""{character_card}{combined_context[:available_tokens]}{npc.name} says:\nPlease respond with structured output in the following format:\n{{"dialogue": "<response>"}}"""

    response = openai.Completion.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=16384
    )
    return {"dialogue": response.choices[0].text.strip()}

# Function to generate emotion
def generate_emotion(npc: NPC, context: Optional[str], memory_context: str) -> dict:
    # Use GPT-4o-mini to generate an emote with intelligent token usage
    max_input_tokens = 128000 - 16384  # Reserve 16384 tokens for output
    character_card = f"Character Card:\nName: {npc.name}\nSpecies: {npc.species}\nAge: {npc.age}\nGender: {npc.gender}\nSkills: {', '.join(npc.skills)}\nBackground: {npc.background}\nAbilities: {', '.join(npc.abilities)}\n"
    combined_context = f"Memories:\n{memory_context}\n\nContext: {context}\n" if context else f"Memories:\n{memory_context}\n"
    available_tokens = max_input_tokens - len(character_card.split()) - len(combined_context.split())
    prompt = f"""{character_card}{combined_context[:available_tokens]}{npc.name} emotes:\nPlease respond with structured output in the following format:\n{{"emote": "<response>"}}"""

    response = openai.Completion.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=16384
    )
    return {"emote": response.choices[0].text.strip()}

# Modify NPC endpoint
@app.post("/npc/{npc_id}/modify")
async def modify_npc(npc_id: str, new_data: NPC, true_name: str):
    npc = npc_storage.get(npc_id)
    if npc is None:
        raise HTTPException(status_code=404, detail="NPC not found")

    if npc.true_name != true_name:
        raise HTTPException(status_code=403, detail="True name does not match. Unauthorized modification.")

    # Update NPC information
    npc_storage[npc_id] = new_data
    return {"npc_id": npc_id, "message": "NPC modified successfully"}

# Delete NPC memory endpoint
@app.delete("/npc/{npc_id}/forget")
async def forget_npc(npc_id: str, true_name: str):
    npc = npc_storage.get(npc_id)
    if npc is None:
        raise HTTPException(status_code=404, detail="NPC not found")

    if npc.true_name != true_name:
        raise HTTPException(status_code=403, detail="True name does not match. Unauthorized modification.")

    # Remove memory from vector database
    db.delete(where={"npc_id": npc_id})
    return {"npc_id": npc_id, "message": "NPC memory deleted successfully"}