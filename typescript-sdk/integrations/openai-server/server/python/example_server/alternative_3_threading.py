import os
import uuid
import uvicorn
import asyncio
import threading
from queue import Queue, Empty
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

app = FastAPI(title="AG-UI OpenAI Server - Threading")

client = OpenAI()

@app.post("/")
async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
    """Threading approach with Queue"""
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)
    
    # Create a thread-safe queue
    event_queue = Queue()
    
    def process_openai_stream():
        """Process OpenAI stream in a separate thread"""
        try:
            # Add start event
            event_queue.put(encoder.encode(
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
            
            print('\n\nProcessing in thread...')
            for i, chunk in enumerate(stream):
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(f"Thread processing chunk {i}: '{content}'", flush=True)
                    
                    # Put event in queue
                    event_queue.put(encoder.encode({
                        "type": EventType.TEXT_MESSAGE_CHUNK,
                        "message_id": message_id,
                        "delta": content,
                    }))
            
            # Add end event
            event_queue.put(encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
            ))
            
            # Signal completion
            event_queue.put(None)  # Sentinel value
            
        except Exception as error:
            event_queue.put(encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=str(error)
                )
            ))
            event_queue.put(None)  # Sentinel value
    
    async def stream_from_queue():
        """Stream events from the queue"""
        # Start the processing thread
        thread = threading.Thread(target=process_openai_stream)
        thread.start()
        
        while True:
            try:
                # Get event from queue (non-blocking)
                event = event_queue.get_nowait()
                
                # Check for completion sentinel
                if event is None:
                    break
                
                print("Streaming event from queue")
                yield event + "\n"
                
            except Empty:
                # No events available, wait a bit
                await asyncio.sleep(0.01)
        
        # Wait for thread to complete
        thread.join()
    
    return StreamingResponse(
        stream_from_queue(),
        media_type=encoder.get_content_type()
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run("alternative_3_threading:app", host="0.0.0.0", port=port, reload=True)
