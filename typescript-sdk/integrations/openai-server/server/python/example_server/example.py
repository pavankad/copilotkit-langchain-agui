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

app = FastAPI(title="AG-UI OpenAI Server")

# Initialize OpenAI client - uses OPENAI_API_KEY from environment
client = OpenAI()

@app.post("/")
async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
    """OpenAI agentic chat endpoint"""
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)

    async def event_generator():
        try:
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
            )
            for message in input_data.messages :
                if message.role == "user":
                    prompt = message.content
                    print(f'User prompt: {prompt}')
                    break
                    
            # Call OpenAI's API with streaming enabled
            stream = client.chat.completions.create(
                model="gpt-4o",
                stream=True,
                # Transform AG-UI messages to OpenAI's message format
                messages=[
                    {
                        "role": message.role,
                        "content": prompt,
                        # Include tool calls if this is an assistant message with tools
                        #**({"tool_calls": message.tool_calls} if message.role == "assistant" and hasattr(message, 'tool_calls') and message.tool_calls else {}),
                        # Include tool call ID if this is a tool result message
                        #**({"tool_call_id": message.tool_call_id} if message.role == "tool" and hasattr(message, 'tool_call_id') else {}),
                    }
                    #for message in input_data.messages
                ],
            )

            message_id = str(uuid.uuid4())

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
                            delta="Hello world!"
                        )
                    )
                # Handle tool call chunks
                elif chunk.choices[0].delta.tool_calls:
                    tool_call = chunk.choices[0].delta.tool_calls[0]

                    yield encoder.encode({
                        "type": EventType.TOOL_CALL_CHUNK,
                        "tool_call_id": tool_call.id,
                        "tool_call_name": tool_call.function.name if tool_call.function else None,
                        "parent_message_id": message_id,
                        "delta": tool_call.function.arguments if tool_call.function else None,
                    })

            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id
                )
            )

        except Exception as error:
            yield encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=str(error)
                )
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

if __name__ == "__main__":
    main()