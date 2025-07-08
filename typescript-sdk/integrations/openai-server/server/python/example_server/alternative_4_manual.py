import os
import uuid
import uvicorn
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

app = FastAPI(title="AG-UI OpenAI Server - Manual Chunking")

client = OpenAI()

@app.post("/")
async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
    """Manual chunking approach without complex generators"""
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)
    
    def create_response_stream():
        """Create response stream without async generator complexity"""
        events = []
        
        try:
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
            
            # Call OpenAI API
            stream = client.chat.completions.create(
                model="gpt-4o",
                stream=True,
                messages=[{"role": "user", "content": prompt}]
            )
            
            message_id = str(uuid.uuid4())
            
            print('\n\nProcessing chunks...')
            for i, chunk in enumerate(stream):
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(f"Processing chunk {i}: '{content}'", flush=True)
                    
                    # Add each chunk event
                    events.append(encoder.encode({
                        "type": EventType.TEXT_MESSAGE_CHUNK,
                        "message_id": message_id,
                        "delta": content,
                    }))
            
            # End event
            events.append(encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
            ))
            
            print(f'\n\nGenerated {len(events)} events total')
            
            # Return all events as a streaming response
            for event in events:
                yield event + "\n"
                
        except Exception as error:
            yield encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=str(error)
                )
            ) + "\n"
    
    return StreamingResponse(
        create_response_stream(),
        media_type=encoder.get_content_type()
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8003"))
    uvicorn.run("alternative_4_manual:app", host="0.0.0.0", port=port, reload=True)
