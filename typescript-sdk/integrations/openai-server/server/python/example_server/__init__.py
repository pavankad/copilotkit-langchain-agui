"""
Example server for the AG-UI protocol.
"""

import os
import uvicorn
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from ag_ui.core import (
    RunAgentInput,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageChunkEvent,
)
from ag_ui.encoder import EventEncoder
import openai

app = FastAPI(title="AG-UI Endpoint")

client = openai.OpenAI()  # Initialize OpenAI client - uses OPENAI_API_KEY from environment

@app.post("/")
async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
    """Agentic chat endpoint"""
    # Get the accept header from the request
    accept_header = request.headers.get("accept")

    # Create an event encoder to properly format SSE events
    encoder = EventEncoder(accept=accept_header)

    async def event_generator():

        # Send run started event
        yield encoder.encode(
          RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id
          ),
        )

        message_id = str(uuid.uuid4())

        yield encoder.encode(
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role="assistant"
            )
        )

       
        # Call OpenAI's API with streaming enabled
        stream = client.chat.completions.create(
            model="gpt-4o",
            stream=True,
            # Convert AG-UI tools format to OpenAI's expected format
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                }
                for tool in input_data.tools
            ] if input_data.tools else None,
            # Transform AG-UI messages to OpenAI's message format
            messages=[
                {
                    "role": message.role,
                    "content": message.content or "",
                    # Include tool calls if this is an assistant message with tools
                    **({"tool_calls": message.tool_calls} if message.role == "assistant" and hasattr(message, 'tool_calls') and message.tool_calls else {}),
                    # Include tool call ID if this is a tool result message
                    **({"tool_call_id": message.tool_call_id} if message.role == "tool" and hasattr(message, 'tool_call_id') else {}),
                }
                for message in input_data.messages
            ],
        )

        # Stream each chunk from OpenAI's response
        print('\n\n Stream response')
        for i, chunk in enumerate(stream):
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(f"Chunk {i}: '{content}'", flush=True)  # More explicit logging
                yield encoder.encode(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=content
                    )
                )

        yield encoder.encode(
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=message_id
            )
        )

        # Send run finished event
        yield encoder.encode(
          RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id
          ),
        )

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type()
    )

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "example_server:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
