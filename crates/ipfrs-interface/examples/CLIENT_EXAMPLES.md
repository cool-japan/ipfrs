# IPFRS HTTP Client Examples

This document provides examples of using the IPFRS HTTP Gateway from various clients.

## Table of Contents

- [curl Examples](#curl-examples)
- [Python Examples](#python-examples)
- [JavaScript Examples](#javascript-examples)
- [Go Examples](#go-examples)

---

## curl Examples

### Health Check

```bash
curl http://localhost:8080/health
```

**Response:**
```json
{"status":"ok"}
```

### Upload a File (Kubo API)

```bash
# Upload a text file
echo "Hello, IPFRS!" > test.txt
curl -X POST -F "file=@test.txt" http://localhost:8080/api/v0/add
```

**Response:**
```json
{
  "Hash": "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
  "Size": 14
}
```

### Download a File (Kubo API)

```bash
# Using the CID from the upload
curl -X POST "http://localhost:8080/api/v0/cat?arg=QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"
```

### Gateway Retrieval (HTTP GET)

```bash
# Retrieve content via gateway
curl "http://localhost:8080/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"
```

### Range Request

```bash
# Download bytes 0-99
curl -H "Range: bytes=0-99" \
     "http://localhost:8080/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"
```

### Block Operations

```bash
# Get raw block
curl -X POST "http://localhost:8080/api/v0/block/get?arg=QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"

# Put raw block
curl -X POST -F "block=@data.bin" http://localhost:8080/api/v0/block/put
```

### Batch Block Operations (v1 API)

```bash
# Batch get blocks
curl -X POST http://localhost:8080/v1/block/batch/get \
  -H "Content-Type: application/json" \
  -d '{
    "cids": [
      "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
      "QmYabcd1234567890"
    ]
  }'
```

### Streaming Upload (v1 API)

```bash
# Upload large file with progress tracking
curl -X POST -F "file=@largefile.bin" \
     http://localhost:8080/v1/stream/upload
```

### Tensor Operations (v1 API)

```bash
# Get tensor metadata
curl http://localhost:8080/v1/tensor/QmTensorCID/info

# Get tensor with slice
curl "http://localhost:8080/v1/tensor/QmTensorCID?slice=0:10,5:15"
```

### Node Information

```bash
# Get node ID
curl -X POST http://localhost:8080/api/v0/id

# Get version
curl -X POST http://localhost:8080/api/v0/version

# Get bandwidth stats
curl -X POST http://localhost:8080/api/v0/stats/bw
```

---

## Python Examples

### Basic Upload and Download

```python
import requests

# Upload a file
url = "http://localhost:8080/api/v0/add"
files = {"file": open("test.txt", "rb")}
response = requests.post(url, files=files)
result = response.json()
cid = result["Hash"]
print(f"Uploaded file with CID: {cid}")

# Download the file
url = f"http://localhost:8080/api/v0/cat?arg={cid}"
response = requests.post(url)
content = response.content
print(f"Downloaded content: {content.decode()}")
```

### Gateway Retrieval with Range

```python
import requests

cid = "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"
url = f"http://localhost:8080/ipfs/{cid}"

# Get bytes 0-99
headers = {"Range": "bytes=0-99"}
response = requests.get(url, headers=headers)

if response.status_code == 206:
    print("Partial content received")
    print(f"Content-Range: {response.headers.get('Content-Range')}")
    print(f"Data: {response.content[:50]}...")
```

### Batch Operations

```python
import requests

url = "http://localhost:8080/v1/block/batch/get"
data = {
    "cids": [
        "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
        "QmYabcd1234567890"
    ]
}

response = requests.post(url, json=data)
result = response.json()

for block in result["blocks"]:
    print(f"CID: {block['cid']}, Size: {len(block['data'])}")
```

### Streaming Upload with Progress

```python
import requests
from tqdm import tqdm

def upload_with_progress(file_path):
    url = "http://localhost:8080/v1/stream/upload"

    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
            def callback(monitor):
                pbar.update(monitor.bytes_read - pbar.n)

            files = {"file": f}
            response = requests.post(url, files=files)

    return response.json()

result = upload_with_progress("largefile.bin")
print(f"Upload complete: {result['cid']}")
```

### Tensor Slicing

```python
import requests
import numpy as np

cid = "QmTensorCID"

# Get tensor info
info_url = f"http://localhost:8080/v1/tensor/{cid}/info"
info = requests.get(info_url).json()
print(f"Tensor shape: {info['shape']}")
print(f"Data type: {info['dtype']}")

# Get a slice
slice_url = f"http://localhost:8080/v1/tensor/{cid}?slice=0:10,5:15"
response = requests.get(slice_url)

# Parse response headers
shape = eval(response.headers['X-Tensor-Shape'])
dtype = response.headers['X-Tensor-Dtype']

# Convert to numpy array
data = np.frombuffer(response.content, dtype=np.float32)
tensor = data.reshape(shape)
print(f"Slice shape: {tensor.shape}")
```

---

## JavaScript Examples

### Node.js with axios

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

// Upload a file
async function uploadFile(filePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));

  const response = await axios.post(
    'http://localhost:8080/api/v0/add',
    form,
    { headers: form.getHeaders() }
  );

  console.log('Uploaded:', response.data);
  return response.data.Hash;
}

// Download a file
async function downloadFile(cid) {
  const response = await axios.post(
    `http://localhost:8080/api/v0/cat?arg=${cid}`,
    {},
    { responseType: 'arraybuffer' }
  );

  return Buffer.from(response.data);
}

// Batch get blocks
async function batchGet(cids) {
  const response = await axios.post(
    'http://localhost:8080/v1/block/batch/get',
    { cids }
  );

  return response.data.blocks;
}

// Example usage
(async () => {
  const cid = await uploadFile('test.txt');
  const content = await downloadFile(cid);
  console.log('Content:', content.toString());

  const blocks = await batchGet([cid]);
  console.log('Blocks:', blocks);
})();
```

### Browser with Fetch API

```javascript
// Upload file from input element
async function uploadFile(fileInput) {
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  const response = await fetch('http://localhost:8080/api/v0/add', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  return result.Hash;
}

// Download and display
async function downloadAndDisplay(cid) {
  const response = await fetch(
    `http://localhost:8080/ipfs/${cid}`
  );

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);

  // Display in img tag or download
  const img = document.createElement('img');
  img.src = url;
  document.body.appendChild(img);
}

// Range request
async function downloadRange(cid, start, end) {
  const response = await fetch(
    `http://localhost:8080/ipfs/${cid}`,
    {
      headers: {
        'Range': `bytes=${start}-${end}`
      }
    }
  );

  if (response.status === 206) {
    const data = await response.arrayBuffer();
    console.log('Partial content:', data);
  }
}
```

### WebSocket for Real-time Updates

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  console.log('Connected to IPFRS WebSocket');

  // Subscribe to block events
  ws.send(JSON.stringify({
    type: 'Subscribe',
    topic: 'blocks'
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'Event') {
    const event = JSON.parse(message.payload);
    console.log('Received event:', event);

    if (event.BlockAdded) {
      console.log('New block added:', event.BlockAdded.cid);
    }
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

---

## Go Examples

### Basic Client

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
)

// Upload a file
func uploadFile(filePath string) (string, error) {
    file, err := os.Open(filePath)
    if err != nil {
        return "", err
    }
    defer file.Close()

    body := &bytes.Buffer{}
    writer := multipart.NewWriter(body)
    part, err := writer.CreateFormFile("file", filePath)
    if err != nil {
        return "", err
    }
    io.Copy(part, file)
    writer.Close()

    resp, err := http.Post(
        "http://localhost:8080/api/v0/add",
        writer.FormDataContentType(),
        body,
    )
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    var result struct {
        Hash string
        Size int
    }
    json.NewDecoder(resp.Body).Decode(&result)
    return result.Hash, nil
}

// Download a file
func downloadFile(cid string) ([]byte, error) {
    url := fmt.Sprintf("http://localhost:8080/api/v0/cat?arg=%s", cid)
    resp, err := http.Post(url, "", nil)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}

func main() {
    cid, err := uploadFile("test.txt")
    if err != nil {
        panic(err)
    }
    fmt.Printf("Uploaded: %s\n", cid)

    content, err := downloadFile(cid)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Content: %s\n", string(content))
}
```

### Batch Operations

```go
type BatchGetRequest struct {
    CIDs []string `json:"cids"`
}

type BatchGetResponse struct {
    Blocks []struct {
        CID  string `json:"cid"`
        Data string `json:"data"`
    } `json:"blocks"`
}

func batchGet(cids []string) (*BatchGetResponse, error) {
    req := BatchGetRequest{CIDs: cids}
    body, _ := json.Marshal(req)

    resp, err := http.Post(
        "http://localhost:8080/v1/block/batch/get",
        "application/json",
        bytes.NewBuffer(body),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result BatchGetResponse
    json.NewDecoder(resp.Body).Decode(&result)
    return &result, nil
}
```

---

## Testing Tips

### Using httpie (User-Friendly Alternative to curl)

```bash
# Install httpie
pip install httpie

# Upload file
http -f POST localhost:8080/api/v0/add file@test.txt

# Download file
http POST localhost:8080/api/v0/cat arg==QmXoyp...

# Batch operation with JSON
http POST localhost:8080/v1/block/batch/get \
  cids:='["QmXoyp...", "QmYabc..."]'
```

### Performance Testing with ab (Apache Bench)

```bash
# Test 1000 requests with 10 concurrent connections
ab -n 1000 -c 10 http://localhost:8080/health
```

### Load Testing with wrk

```bash
# Run for 30 seconds with 12 threads and 400 connections
wrk -t12 -c400 -d30s http://localhost:8080/health
```

---

## Error Handling

All endpoints return JSON error responses with this format:

```json
{
  "error": "Content not found",
  "code": "NOT_FOUND",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Common HTTP status codes:
- `200 OK` - Success
- `206 Partial Content` - Range request success
- `304 Not Modified` - Cached content still valid
- `400 Bad Request` - Invalid input
- `404 Not Found` - Content not found
- `416 Range Not Satisfiable` - Invalid range
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

---

## Authentication

For endpoints that require authentication, include a Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:8080/api/v0/add
```

Or use an API key:

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
     http://localhost:8080/api/v0/add
```

---

## Additional Resources

- [OpenAPI Specification](../openapi.yaml) - Full API documentation
- [Configuration Guide](../CONFIGURATION.md) - Server configuration options
- [IPFS HTTP API Reference](https://docs.ipfs.tech/reference/kubo/rpc/) - Kubo compatibility reference
