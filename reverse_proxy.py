from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import grpc
import grpc.aio
import json
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import deserialize_bytes_tensor
import uvicorn
from time import time

app = FastAPI()

@app.post("/v1/completions")
async def completions(request: Request):
    data = await request.json()
    
    # Extract parameters from the request
    prompt = data.get("prompt")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.95)
    stream = data.get("stream", False)

    if not prompt:
        raise HTTPException(status_code=400, detail="The 'prompt' field is required.")
    
    model_name = "vllm-qwen2_5-coder-7b"
    model_version = ""
    
    # Create asynchronous gRPC channel and stub
    channel = grpc.aio.insecure_channel("localhost:8001")
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    
    # Create the Triton request
    triton_request = service_pb2.ModelInferRequest()
    triton_request.model_name = model_name
    triton_request.model_version = model_version
    triton_request.id = "my_request_id"
    
    # Input 1: text_input
    input_text = triton_request.InferInputTensor()
    input_text.name = "text_input"
    input_text.datatype = "BYTES"
    input_text.shape.extend([1])
    input_text.contents.bytes_contents.append(prompt.encode('utf-8'))
    
    # Input 2: stream
    input_stream = triton_request.InferInputTensor()
    input_stream.name = "stream"
    input_stream.datatype = "BOOL"
    input_stream.shape.extend([1])
    input_stream.contents.bool_contents.append(stream)
    
    # Input 3: sampling_parameters
    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    input_sampling = triton_request.InferInputTensor()
    input_sampling.name = "sampling_parameters"
    input_sampling.datatype = "BYTES"
    input_sampling.shape.extend([1])
    input_sampling.contents.bytes_contents.append(
        json.dumps(sampling_params).encode('utf-8')
    )
    
    # Input 4: exclude_input_in_output
    input_exclude = triton_request.InferInputTensor()
    input_exclude.name = "exclude_input_in_output"
    input_exclude.datatype = "BOOL"
    input_exclude.shape.extend([1])
    input_exclude.contents.bool_contents.append(True)
    
    # Add all inputs to the request
    triton_request.inputs.extend([input_text, input_stream, input_sampling, input_exclude])
    
    # Output
    output = triton_request.InferRequestedOutputTensor()
    output.name = "text_output"
    triton_request.outputs.extend([output])
    
    # Asynchronous request iterator
    async def request_iterator():
        yield triton_request
    
    # Function to handle streaming responses
    async def stream_response():
        responses = grpc_stub.ModelStreamInfer(request_iterator())
        async for response in responses:
            if response.HasField('infer_response'):
                infer_response = response.infer_response
                # Deserialize the output tensor
                for raw_output in infer_response.raw_output_contents:
                    text_outputs = deserialize_bytes_tensor(raw_output)
                    for text_output in text_outputs:
                        # Build the OpenAI API response format
                        completion_response = {
                            "choices": [
                                {
                                    "text": text_output.decode('utf-8'),
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None
                                }
                            ],
                            "id": "cmpl-123",
                            "object": "text_completion",
                            "created": int(time()),
                            "model": model_name,
                        }
                        print(json.dumps(completion_response))
                        yield f"data: {json.dumps(completion_response)}\n\n"
            elif response.HasField('error'):
                error_message = response.error.message
                raise HTTPException(status_code=500, detail=f"Triton error: {error_message}")
        # Signal the end of the stream
        yield "data: [DONE]\n\n"
    
    if stream:
        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        # Non-streaming response
        responses = grpc_stub.ModelStreamInfer(request_iterator())
        full_output = ""
        async for response in responses:
            if response.HasField('infer_response'):
                infer_response = response.infer_response
                # Deserialize the output tensor
                raw_output = infer_response.raw_output_contents[0]
                text_outputs = deserialize_bytes_tensor(raw_output)
                for text_output in text_outputs:
                    full_output += text_output.decode('utf-8')
            elif response.HasField('error'):
                error_message = response.error.message
                raise HTTPException(status_code=500, detail=f"Triton error: {error_message}")
        
        completion_response = {
            "choices": [
                {
                    "text": full_output,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None
                }
            ],
            "id": "cmpl-123",
            "object": "text_completion",
            "created": int(time()),
            "model": model_name,
        }
        return JSONResponse(content=completion_response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000)
