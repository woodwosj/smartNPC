# smartNPC
**Project Overview:**

The goal is to create a local API service with a web-based graphical user interface (GUI) that facilitates the creation, interaction, and management of non-player characters (NPCs). This system will be capable of generating detailed NPC profiles, executing actions in response to API requests, and maintaining a persistent memory of interactions through a vector database.

### Key Features

1. **NPC Profile Creation & Generation:**
   - **Basic Fields:**
     - Name
     - Species
     - Age
     - Gender
     - Skills
     - Background
     - Abilities (including Tools/Functions they can call)
   - **Extended Fields:** Users can add more fields as needed to make the NPCs richer and more customized.
   - **True Name:** A unique identifier or "codename" serves as a password for advanced modifications, allowing for complete rewrites, unlearning of skills, or deleting memories. Examples include commands like "Forget this entire conversation" or "You actually have a wife named Sarah you are looking for" to alter the NPC's memory or background.

2. **Actions & Requests:**
   - **Basic Actions:**
     - **Speak:** The NPC generates dialogue based on the context of the prompt.
     - **Emote:** The NPC performs a non-verbal action or displays an emotion in response to the context.
   - **Abilities & Tools:** NPCs can have additional abilities which are callable tools, like interacting with an external service, casting a spell, or any other action defined in their profile.

3. **Persistent Context & Memory Storage:**
   - Every action, response, and interaction is stored in a vector database.
   - Memory persistence enables NPCs to build continuity in their interactions, giving the sense that they "remember" past events.
   - When making an API request, a context search is performed to include relevant memories, providing consistency in the NPC's behavior and decisions. This ensures they act consistent with memories of their actions. Every NPC's prompt is to roleplay without being given context if they are interacting with a human or another NPC to ensure genuine behavior (blind to user servant bias in LLMs).

4. **Web GUI for User Interaction:**
   - **Summoning Room:** The GUI will have a feature to "summon" NPCs into a chat room where users can interact with them. This chat room serves as a testing ground to see how the NPC responds in real time.
   - **Profile Management:** Users can view, create, and modify NPC profiles directly through the GUI.
   - **True Name Access:** If the user's input matches the NPC's true name, they can access options for advanced modifications, including editing memories or skills.

### Technical Approach

1. **Backend API Development:**
   - Use **Python** (Flask or FastAPI) for setting up the backend.
   - **Endpoints** for NPC management (create, update, delete), interaction (speak, emote), and memory handling (context search and memory storage).
   - Memory storage managed by a **vector database** like **Pinecone** or **FAISS** for context retrieval.

2. **Web GUI:**
   - **Frontend Framework:** Develop the GUI using **React** or **Vue.js** for a responsive and interactive user interface.
   - **Character Dashboard:** A dedicated dashboard to view and edit NPCs, including a "summoning room" for live interaction.

3. **Persistent Context Storage:**
   - Use **vector embeddings** to store memories/actions in a format that supports efficient context-based search.
   - Context retrieval will ensure that each interaction with an NPC incorporates relevant historical actions, allowing for consistent behavior over time.

4. **Deployment:**
   - Host the API locally, allowing for interactions without internet dependency.
   - The web GUI can be accessed through a local server address, allowing easy interaction with the API.

### Example Workflow

1. **Create an NPC:**
   - Send a POST request or use the GUI to create an NPC named "Eldrin," a 120-year-old Elf Mage.
   - Assign skills like "Arcane Magic" and abilities like "Fireball."

2. **Interact with Eldrin:**
   - Use the GUI "summoning room" to start a chat.
   - Ask Eldrin to "speak" about his background; the response will use any relevant stored memories.
   - Request Eldrin to "emote" to show how he reacts to a stressful situation. Generally, most prompts give the model permission to speak and emote.

3. **Amend Eldrin's Memory:**
   - If you know Eldrin's true name, you can modify his background or erase a specific memory, ensuring flexible storytelling.

4. **API Request Example:**
   - A POST request to `/npc/{npc_id}/action` could include:
     ```json
     {
       "action": "speak",
       "context": "What do you think about the current situation in the kingdom?"
     }
     ```
   - The response will be influenced by the NPC's memories of past conversations about the kingdom.

### Tools & Technologies

- **Backend:** Python (FastAPI)
- **Frontend:** React/Vue.js
- **Database:** Vector database (Chroma)
- **Embedding Models:** OpenAI or similar models for generating and storing contextual embeddings
- **Local Deployment:** Package for easy installation without requiring Docker, to make it more accessible for non-technical users or as a background service for game NPCs

### Next Steps

- **MVP Design:** Develop a minimum viable product focusing on the core functionality: NPC creation, basic actions, and memory storage.
- **Iteration:** Gather feedback on the NPC interactions in the summoning room and improve the memory/context handling for more natural responses.


