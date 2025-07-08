import os
import uuid
import uvicorn
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from ag_ui.core import (
    RunAgentInput,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
)
from ag_ui.encoder import EventEncoder
from openai import OpenAI
from collections import deque

app = FastAPI(title="AG-UI OpenAI Server - Queue Buffer")

client = OpenAI()

@app.post("/")
async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
    """Queue-based approach with manual buffer management"""
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)
    
    # Create a buffer for events
    event_buffer = deque()
    
    async def buffer_events():
        """Process OpenAI stream and add to buffer"""
        try:
            # Add start event to buffer
            event_buffer.append(encoder.encode(
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
            
            # Call OpenAI API
            stream = client.chat.completions.create(
                model="gpt-4o",
                stream=True,
                messages=[{"role": "user", "content": prompt}]
            )
            
            message_id = str(uuid.uuid4())
            
            print('\n\nStreaming to buffer...')
            for i, chunk in enumerate(stream):
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(f"Buffering chunk {i}: '{content}'", flush=True)
                    
                    # Add to buffer instead of yielding
                    event_buffer.append(encoder.encode({
                        "type": EventType.TEXT_MESSAGE_CHUNK,
                        "message_id": message_id,
                        "delta": content,
                    }))
                    
                    # Small delay to simulate processing
                    await asyncio.sleep(0.01)
            
            # Add end event
            event_buffer.append(encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
            ))
            
        except Exception as error:
            event_buffer.append(encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=str(error)
                )
            ))
    
    async def stream_from_buffer():
        """Stream events from the buffer"""
        # Start the buffering process
        buffer_task = asyncio.create_task(buffer_events())
        
        # Stream events as they become available
        while True:
            # Send any buffered events
            while event_buffer:
                event = event_buffer.popleft()
                print("Streaming event from buffer")
                yield event + "\n"  # Add newline for SSE format
                await asyncio.sleep(0.1)  # Control streaming rate
            
            # Check if buffering is complete
            if buffer_task.done():
                break
            
            # Wait a bit before checking buffer again
            await asyncio.sleep(0.05)
    
    return StreamingResponse(
        stream_from_buffer(),
        media_type=encoder.get_content_type()
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("alternative_2_queue:app", host="0.0.0.0", port=port, reload=True)
