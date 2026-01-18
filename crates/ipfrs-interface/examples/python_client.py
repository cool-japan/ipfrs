#!/usr/bin/env python3
"""
IPFRS Python Client Example

This example demonstrates how to interact with the IPFRS HTTP Gateway API
using Python, including:
- File upload and download
- Batch operations
- Tensor operations with Apache Arrow support
- Streaming uploads/downloads
- WebSocket real-time events
"""

import requests
import json
import pyarrow as pa
import pyarrow.ipc as ipc
from typing import List, Dict, Optional, BinaryIO
import io

class IPFRSClient:
    """Simple IPFRS HTTP client implementation"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize IPFRS client

        Args:
            base_url: Base URL of the IPFRS gateway
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    # ========================================================================
    # Kubo v0 API - IPFS Compatibility
    # ========================================================================

    def add_file(self, file_path: str) -> Dict[str, any]:
        """
        Upload a file to IPFRS (Kubo v0 API)

        Args:
            file_path: Path to file to upload

        Returns:
            Dict with 'Hash' (CID) and 'Size'
        """
        url = f"{self.base_url}/api/v0/add"
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(url, files=files)
            response.raise_for_status()
            return response.json()

    def cat(self, cid: str) -> bytes:
        """
        Download a file from IPFRS (Kubo v0 API)

        Args:
            cid: Content Identifier

        Returns:
            File content as bytes
        """
        url = f"{self.base_url}/api/v0/cat"
        response = self.session.post(url, params={'arg': cid})
        response.raise_for_status()
        return response.content

    def get_block(self, cid: str) -> bytes:
        """
        Get raw block data

        Args:
            cid: Content Identifier

        Returns:
            Raw block bytes
        """
        url = f"{self.base_url}/api/v0/block/get"
        response = self.session.post(url, params={'arg': cid})
        response.raise_for_status()
        return response.content

    def put_block(self, data: bytes) -> Dict[str, any]:
        """
        Store raw block data

        Args:
            data: Raw block bytes

        Returns:
            Dict with 'Hash' and 'Size'
        """
        url = f"{self.base_url}/api/v0/block/put"
        files = {'block': io.BytesIO(data)}
        response = self.session.post(url, files=files)
        response.raise_for_status()
        return response.json()

    # ========================================================================
    # Gateway API - HTTP GET
    # ========================================================================

    def get(self, cid: str, byte_range: Optional[tuple] = None) -> bytes:
        """
        Get content via HTTP gateway

        Args:
            cid: Content Identifier
            byte_range: Optional (start, end) tuple for range request

        Returns:
            Content bytes
        """
        url = f"{self.base_url}/ipfs/{cid}"
        headers = {}

        if byte_range:
            start, end = byte_range
            headers['Range'] = f'bytes={start}-{end}'

        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.content

    # ========================================================================
    # High-Speed v1 API - Batch Operations
    # ========================================================================

    def batch_get_blocks(self, cids: List[str]) -> List[Dict[str, any]]:
        """
        Retrieve multiple blocks in parallel

        Args:
            cids: List of Content Identifiers

        Returns:
            List of dicts with 'cid' and 'data' (base64)
        """
        url = f"{self.base_url}/v1/block/batch/get"
        payload = {'cids': cids}
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()['blocks']

    def batch_has_blocks(self, cids: List[str]) -> List[Dict[str, any]]:
        """
        Check existence of multiple blocks

        Args:
            cids: List of Content Identifiers

        Returns:
            List of dicts with 'cid' and 'exists'
        """
        url = f"{self.base_url}/v1/block/batch/has"
        payload = {'cids': cids}
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()['results']

    # ========================================================================
    # Streaming API
    # ========================================================================

    def streaming_upload(self, file_path: str) -> Dict[str, any]:
        """
        Upload large file with streaming

        Args:
            file_path: Path to file to upload

        Returns:
            Dict with 'cid', 'size', 'chunks_received'
        """
        url = f"{self.base_url}/v1/stream/upload"
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(url, files=files)
            response.raise_for_status()
            return response.json()

    def streaming_download(self, cid: str, chunk_size: int = 65536) -> bytes:
        """
        Download content with streaming

        Args:
            cid: Content Identifier
            chunk_size: Chunk size in bytes

        Returns:
            Complete content
        """
        url = f"{self.base_url}/v1/stream/download/{cid}"
        params = {'chunk_size': chunk_size}
        response = self.session.get(url, params=params, stream=True)
        response.raise_for_status()

        chunks = []
        for chunk in response.iter_content(chunk_size=chunk_size):
            chunks.append(chunk)

        return b''.join(chunks)

    # ========================================================================
    # Tensor API - Zero-Copy with Arrow Support
    # ========================================================================

    def get_tensor(self, cid: str, slice_spec: Optional[str] = None) -> bytes:
        """
        Get tensor data (raw format)

        Args:
            cid: Content Identifier
            slice_spec: Optional slice specification (e.g., "0:10,5:15")

        Returns:
            Raw tensor bytes
        """
        url = f"{self.base_url}/v1/tensor/{cid}"
        params = {}
        if slice_spec:
            params['slice'] = slice_spec

        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.content

    def get_tensor_info(self, cid: str) -> Dict[str, any]:
        """
        Get tensor metadata only

        Args:
            cid: Content Identifier

        Returns:
            Dict with tensor metadata
        """
        url = f"{self.base_url}/v1/tensor/{cid}/info"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_tensor_arrow(self, cid: str, slice_spec: Optional[str] = None) -> pa.Table:
        """
        Get tensor as Apache Arrow table

        Args:
            cid: Content Identifier
            slice_spec: Optional slice specification

        Returns:
            PyArrow Table with tensor data
        """
        url = f"{self.base_url}/v1/tensor/{cid}/arrow"
        params = {}
        if slice_spec:
            params['slice'] = slice_spec

        response = self.session.get(url, params=params)
        response.raise_for_status()

        # Parse Arrow IPC stream
        reader = ipc.open_stream(response.content)
        table = reader.read_all()

        # Extract metadata from headers
        shape_str = response.headers.get('X-Tensor-Shape', '[]')
        dtype_str = response.headers.get('X-Tensor-Dtype', 'unknown')

        # Add metadata to table
        metadata = {
            'tensor_shape': shape_str,
            'tensor_dtype': dtype_str,
            'tensor_elements': response.headers.get('X-Tensor-Elements', '0')
        }

        return table, metadata

    # ========================================================================
    # Node Information
    # ========================================================================

    def get_id(self) -> Dict[str, any]:
        """Get node identity information"""
        url = f"{self.base_url}/api/v0/id"
        response = self.session.post(url)
        response.raise_for_status()
        return response.json()

    def get_version(self) -> Dict[str, any]:
        """Get version information"""
        url = f"{self.base_url}/api/v0/version"
        response = self.session.post(url)
        response.raise_for_status()
        return response.json()

    def get_peers(self) -> List[Dict[str, any]]:
        """Get list of connected peers"""
        url = f"{self.base_url}/api/v0/swarm/peers"
        response = self.session.post(url)
        response.raise_for_status()
        return response.json().get('Peers', [])

    def get_bandwidth_stats(self) -> Dict[str, any]:
        """Get bandwidth statistics"""
        url = f"{self.base_url}/api/v0/stats/bw"
        response = self.session.post(url)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of IPFRS client"""

    # Initialize client
    client = IPFRSClient("http://localhost:8080")

    print("=== IPFRS Python Client Example ===\n")

    # Example 1: Upload and download a file
    print("1. File Upload and Download")
    try:
        # Create a test file
        test_file = "/tmp/test_ipfrs.txt"
        with open(test_file, 'w') as f:
            f.write("Hello, IPFRS! This is a test file.")

        # Upload
        result = client.add_file(test_file)
        cid = result['Hash']
        print(f"   Uploaded file: {cid} ({result['Size']} bytes)")

        # Download
        content = client.cat(cid)
        print(f"   Downloaded: {content.decode()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 2: Batch operations
    print("\n2. Batch Block Operations")
    try:
        # Check if blocks exist
        cids_to_check = [cid]  # Using CID from previous example
        results = client.batch_has_blocks(cids_to_check)
        for result in results:
            print(f"   {result['cid']}: exists={result['exists']}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 3: Tensor with Arrow (if you have a tensor)
    print("\n3. Tensor Operations with Apache Arrow")
    try:
        # This would work if you have a tensor CID
        # tensor_cid = "QmYourTensorCID"
        # table, metadata = client.get_tensor_arrow(tensor_cid)
        # print(f"   Tensor shape: {metadata['tensor_shape']}")
        # print(f"   Tensor dtype: {metadata['tensor_dtype']}")
        # df = table.to_pandas()
        # print(f"   As Pandas DataFrame:\n{df.head()}")
        print("   (Skipping - no tensor CID available)")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 4: Node information
    print("\n4. Node Information")
    try:
        version = client.get_version()
        print(f"   Version: {version.get('Version', 'unknown')}")
        print(f"   System: {version.get('System', 'unknown')}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
