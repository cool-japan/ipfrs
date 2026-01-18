#!/usr/bin/env python3
"""
IPFRS gRPC Python Client Example

This example demonstrates how to use the IPFRS gRPC API from Python.

Prerequisites:
    pip install grpcio grpcio-tools

Generate Python proto files:
    python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. \
        ../proto/block.proto ../proto/dag.proto ../proto/file.proto ../proto/tensor.proto

Usage:
    # Start gRPC server first
    cargo run --example grpc_server

    # Then run this client
    python3 examples/grpc_python_client.py
"""

import sys
import grpc

# Note: You need to generate the proto files first (see docstring above)
# This is a template - adjust imports based on generated files
try:
    # These imports will work after generating proto files
    from ipfrs.block.v1 import block_pb2, block_pb2_grpc
except ImportError:
    print("Error: gRPC proto files not generated.")
    print("\nPlease run:")
    print("  python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. \\")
    print("      ../proto/block.proto ../proto/dag.proto ../proto/file.proto ../proto/tensor.proto")
    sys.exit(1)


def run():
    """Run gRPC client examples"""
    print("Connecting to IPFRS gRPC Server...")

    # Connect to server
    channel = grpc.insecure_channel('[::1]:50051')
    stub = block_pb2_grpc.BlockServiceStub(channel)

    print("✓ Connected to server\n")

    # Example 1: Put a block
    print("Example 1: Storing a block")
    data = b"Hello, IPFRS gRPC from Python!"
    request = block_pb2.PutBlockRequest(data=data, cid="")

    response = stub.PutBlock(request)
    if response.HasField('error'):
        print(f"Error: {response.error.message}")
        return

    cid = response.cid
    print(f"  ✓ Block stored with CID: {cid}")
    print(f"  Size: {len(data)} bytes\n")

    # Example 2: Check if block exists
    print("Example 2: Checking if block exists")
    request = block_pb2.HasBlockRequest(cid=cid)
    response = stub.HasBlock(request)

    if response.HasField('error'):
        print(f"Error: {response.error.message}")
        return

    print(f"  ✓ Block exists: {response.exists}\n")

    # Example 3: Retrieve the block
    print("Example 3: Retrieving the block")
    request = block_pb2.GetBlockRequest(cid=cid)
    response = stub.GetBlock(request)

    if response.HasField('error'):
        print(f"Error: {response.error.message}")
        return

    retrieved_data = response.data
    print(f"  ✓ Retrieved data: {retrieved_data.decode('utf-8')}")
    print(f"  CID: {response.cid}")
    print(f"  Size: {response.size} bytes\n")

    # Verify data matches
    if retrieved_data == data:
        print("✓ Success! Data matches original")
    else:
        print("✗ Error: Data mismatch")

    # Example 4: Batch operations
    print("\nExample 4: Batch operations")
    cids = [cid]  # Add more CIDs if available
    request = block_pb2.BatchGetBlocksRequest(cids=cids)

    # Server streaming response
    print("  Retrieving blocks in batch...")
    for response in stub.BatchGetBlocks(request):
        if response.HasField('error'):
            print(f"  Error: {response.error.message}")
            continue
        print(f"  ✓ Retrieved CID: {response.cid} ({response.size} bytes)")

    print("\n✓ All examples completed successfully!")


if __name__ == '__main__':
    run()
