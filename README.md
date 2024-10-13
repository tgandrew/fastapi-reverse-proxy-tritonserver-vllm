# FastAPI Reverse Proxy for Triton Server with vLLM backend

The vLLM Backend makes using Triton Server very simple. The problem is Triton Server doesn't give you an OpenAI
compatible API to work with.

This reverse proxy sets up a minimal example of an OpenAI compatibile API and then forwards the requsts via gRPC to
Triton server for inference.
