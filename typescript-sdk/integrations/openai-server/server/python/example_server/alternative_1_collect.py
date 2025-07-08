import os
import uuid
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ag_ui.core import (
    RunAgentInput,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
)
from ag_ui.encoder import EventEncoder
from openai import OpenAI

app = FastAPI(title="AG-UI OpenAI Server - Non-Streaming")

# Initialize OpenAI client
client = OpenAI()

@app.post("/")
async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
    """Non-streaming approach - collect all content first"""
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)
    
    try:
        # Collect all events in a list
        events = []
        
        # Start event
        events.append(encoder.encode(
            RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id
            )
        ))
        
        # Get user prompt
        prompt = ""
        for message in input_data.messages:
            if message.role == "user":
                prompt = message.content
                break
        
        print(f'User prompt: {prompt}')
        
        # Call OpenAI API with streaming
        stream = client.chat.completions.create(
            model="gpt-4o",
            stream=True,
            messages=[{"role": "user", "content": prompt}]
        )
        
        message_id = str(uuid.uuid4())
        full_content = ""
        
        # Collect all chunks
        print('\n\nCollecting response...')
        for i, chunk in enumerate(stream):
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                print(f"Chunk {i}: '{content}'", flush=True)
        
        print(f'\n\nFull response: {full_content}')
        
        # Add the complete message as a single event
        events.append(encoder.encode({
            "type": EventType.TEXT_MESSAGE_CHUNK,
            "message_id": message_id,
            "delta": full_content,
        }))
        
        # End event
        events.append(encoder.encode(
            RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id
            )
        ))
        
        # Return all events at once
        return JSONResponse(content={"events": events})
        
    except Exception as error:
        return JSONResponse(
            content={"error": str(error)},
            status_code=500
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("alternative_1_collect:app", host="0.0.0.0", port=port, reload=True)
