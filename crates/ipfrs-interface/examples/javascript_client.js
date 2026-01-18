#!/usr/bin/env node
/**
 * IPFRS JavaScript/Node.js Client Example
 *
 * This example demonstrates how to interact with the IPFRS HTTP Gateway API
 * using JavaScript/Node.js, including:
 * - File upload and download
 * - Batch operations
 * - Tensor operations with Apache Arrow support
 * - Streaming uploads/downloads
 * - WebSocket real-time events
 *
 * Dependencies:
 *   npm install axios form-data apache-arrow ws
 */

const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const { tableFromIPC } = require('apache-arrow');
const WebSocket = require('ws');

class IPFRSClient {
    /**
     * Initialize IPFRS client
     * @param {string} baseUrl - Base URL of the IPFRS gateway
     */
    constructor(baseUrl = 'http://localhost:8080') {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: 30000,
        });
    }

    // ========================================================================
    // Kubo v0 API - IPFS Compatibility
    // ========================================================================

    /**
     * Upload a file to IPFRS (Kubo v0 API)
     * @param {string} filePath - Path to file to upload
     * @returns {Promise<{Hash: string, Size: number}>}
     */
    async addFile(filePath) {
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));

        const response = await this.client.post('/api/v0/add', form, {
            headers: form.getHeaders(),
        });

        return response.data;
    }

    /**
     * Download a file from IPFRS (Kubo v0 API)
     * @param {string} cid - Content Identifier
     * @returns {Promise<Buffer>}
     */
    async cat(cid) {
        const response = await this.client.post(`/api/v0/cat?arg=${cid}`, null, {
            responseType: 'arraybuffer',
        });

        return Buffer.from(response.data);
    }

    /**
     * Get raw block data
     * @param {string} cid - Content Identifier
     * @returns {Promise<Buffer>}
     */
    async getBlock(cid) {
        const response = await this.client.post(`/api/v0/block/get?arg=${cid}`, null, {
            responseType: 'arraybuffer',
        });

        return Buffer.from(response.data);
    }

    /**
     * Store raw block data
     * @param {Buffer} data - Raw block bytes
     * @returns {Promise<{Hash: string, Size: number}>}
     */
    async putBlock(data) {
        const form = new FormData();
        form.append('block', data);

        const response = await this.client.post('/api/v0/block/put', form, {
            headers: form.getHeaders(),
        });

        return response.data;
    }

    // ========================================================================
    // Gateway API - HTTP GET
    // ========================================================================

    /**
     * Get content via HTTP gateway
     * @param {string} cid - Content Identifier
     * @param {[number, number]|null} byteRange - Optional [start, end] for range request
     * @returns {Promise<Buffer>}
     */
    async get(cid, byteRange = null) {
        const headers = {};

        if (byteRange) {
            const [start, end] = byteRange;
            headers['Range'] = `bytes=${start}-${end}`;
        }

        const response = await this.client.get(`/ipfs/${cid}`, {
            headers,
            responseType: 'arraybuffer',
        });

        return Buffer.from(response.data);
    }

    // ========================================================================
    // High-Speed v1 API - Batch Operations
    // ========================================================================

    /**
     * Retrieve multiple blocks in parallel
     * @param {string[]} cids - Array of Content Identifiers
     * @returns {Promise<Array<{cid: string, data: string}>>}
     */
    async batchGetBlocks(cids) {
        const response = await this.client.post('/v1/block/batch/get', { cids });
        return response.data.blocks;
    }

    /**
     * Check existence of multiple blocks
     * @param {string[]} cids - Array of Content Identifiers
     * @returns {Promise<Array<{cid: string, exists: boolean}>>}
     */
    async batchHasBlocks(cids) {
        const response = await this.client.post('/v1/block/batch/has', { cids });
        return response.data.results;
    }

    // ========================================================================
    // Streaming API
    // ========================================================================

    /**
     * Upload large file with streaming
     * @param {string} filePath - Path to file to upload
     * @returns {Promise<{cid: string, size: number, chunks_received: number}>}
     */
    async streamingUpload(filePath) {
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));

        const response = await this.client.post('/v1/stream/upload', form, {
            headers: form.getHeaders(),
        });

        return response.data;
    }

    /**
     * Download content with streaming
     * @param {string} cid - Content Identifier
     * @param {number} chunkSize - Chunk size in bytes
     * @returns {Promise<Buffer>}
     */
    async streamingDownload(cid, chunkSize = 65536) {
        const response = await this.client.get(`/v1/stream/download/${cid}`, {
            params: { chunk_size: chunkSize },
            responseType: 'stream',
        });

        const chunks = [];
        for await (const chunk of response.data) {
            chunks.push(chunk);
        }

        return Buffer.concat(chunks);
    }

    // ========================================================================
    // Tensor API - Zero-Copy with Arrow Support
    // ========================================================================

    /**
     * Get tensor data (raw format)
     * @param {string} cid - Content Identifier
     * @param {string|null} sliceSpec - Optional slice specification (e.g., "0:10,5:15")
     * @returns {Promise<Buffer>}
     */
    async getTensor(cid, sliceSpec = null) {
        const params = {};
        if (sliceSpec) {
            params.slice = sliceSpec;
        }

        const response = await this.client.get(`/v1/tensor/${cid}`, {
            params,
            responseType: 'arraybuffer',
        });

        return Buffer.from(response.data);
    }

    /**
     * Get tensor metadata only
     * @param {string} cid - Content Identifier
     * @returns {Promise<Object>}
     */
    async getTensorInfo(cid) {
        const response = await this.client.get(`/v1/tensor/${cid}/info`);
        return response.data;
    }

    /**
     * Get tensor as Apache Arrow table
     * @param {string} cid - Content Identifier
     * @param {string|null} sliceSpec - Optional slice specification
     * @returns {Promise<{table: any, metadata: Object}>}
     */
    async getTensorArrow(cid, sliceSpec = null) {
        const params = {};
        if (sliceSpec) {
            params.slice = sliceSpec;
        }

        const response = await this.client.get(`/v1/tensor/${cid}/arrow`, {
            params,
            responseType: 'arraybuffer',
        });

        // Parse Arrow IPC stream
        const table = tableFromIPC(response.data);

        // Extract metadata from headers
        const metadata = {
            tensor_shape: response.headers['x-tensor-shape'] || '[]',
            tensor_dtype: response.headers['x-tensor-dtype'] || 'unknown',
            tensor_elements: response.headers['x-tensor-elements'] || '0',
        };

        return { table, metadata };
    }

    // ========================================================================
    // Node Information
    // ========================================================================

    /**
     * Get node identity information
     * @returns {Promise<Object>}
     */
    async getId() {
        const response = await this.client.post('/api/v0/id');
        return response.data;
    }

    /**
     * Get version information
     * @returns {Promise<Object>}
     */
    async getVersion() {
        const response = await this.client.post('/api/v0/version');
        return response.data;
    }

    /**
     * Get list of connected peers
     * @returns {Promise<Array>}
     */
    async getPeers() {
        const response = await this.client.post('/api/v0/swarm/peers');
        return response.data.Peers || [];
    }

    /**
     * Get bandwidth statistics
     * @returns {Promise<Object>}
     */
    async getBandwidthStats() {
        const response = await this.client.post('/api/v0/stats/bw');
        return response.data;
    }

    // ========================================================================
    // WebSocket - Real-time Events
    // ========================================================================

    /**
     * Connect to WebSocket for real-time events
     * @param {Function} onMessage - Callback for messages
     * @param {Function} onError - Callback for errors
     * @returns {WebSocket}
     */
    connectWebSocket(onMessage, onError) {
        const wsUrl = this.baseUrl.replace(/^http/, 'ws') + '/ws';
        const ws = new WebSocket(wsUrl);

        ws.on('open', () => {
            console.log('WebSocket connected');
        });

        ws.on('message', (data) => {
            const message = JSON.parse(data.toString());
            onMessage(message);
        });

        ws.on('error', (error) => {
            if (onError) {
                onError(error);
            }
        });

        return ws;
    }

    /**
     * Subscribe to a topic via WebSocket
     * @param {WebSocket} ws - WebSocket connection
     * @param {string} topic - Topic to subscribe to (e.g., "blocks", "peers", "dht")
     */
    subscribe(ws, topic) {
        const message = {
            type: 'Subscribe',
            topic: topic,
        };
        ws.send(JSON.stringify(message));
    }
}


// ============================================================================
// Example Usage
// ============================================================================

async function main() {
    console.log('=== IPFRS JavaScript Client Example ===\n');

    // Initialize client
    const client = new IPFRSClient('http://localhost:8080');

    // Example 1: Upload and download a file
    console.log('1. File Upload and Download');
    try {
        // Create a test file
        const testFile = '/tmp/test_ipfrs_js.txt';
        fs.writeFileSync(testFile, 'Hello, IPFRS! This is a test file from JavaScript.');

        // Upload
        const result = await client.addFile(testFile);
        const cid = result.Hash;
        console.log(`   Uploaded file: ${cid} (${result.Size} bytes)`);

        // Download
        const content = await client.cat(cid);
        console.log(`   Downloaded: ${content.toString()}`);
    } catch (error) {
        console.error(`   Error: ${error.message}`);
    }

    // Example 2: Batch operations
    console.log('\n2. Batch Block Operations');
    try {
        // Using CID from previous example (you'd need to have actual CIDs)
        // const cidsToCheck = [cid];
        // const results = await client.batchHasBlocks(cidsToCheck);
        // results.forEach(result => {
        //     console.log(`   ${result.cid}: exists=${result.exists}`);
        // });
        console.log('   (Skipping - would need valid CIDs)');
    } catch (error) {
        console.error(`   Error: ${error.message}`);
    }

    // Example 3: Tensor with Arrow
    console.log('\n3. Tensor Operations with Apache Arrow');
    try {
        // This would work if you have a tensor CID
        // const tensorCid = 'QmYourTensorCID';
        // const { table, metadata } = await client.getTensorArrow(tensorCid);
        // console.log(`   Tensor shape: ${metadata.tensor_shape}`);
        // console.log(`   Tensor dtype: ${metadata.tensor_dtype}`);
        // console.log(`   Table rows: ${table.numRows}`);
        console.log('   (Skipping - no tensor CID available)');
    } catch (error) {
        console.error(`   Error: ${error.message}`);
    }

    // Example 4: Node information
    console.log('\n4. Node Information');
    try {
        const version = await client.getVersion();
        console.log(`   Version: ${version.Version || 'unknown'}`);
        console.log(`   System: ${version.System || 'unknown'}`);
    } catch (error) {
        console.error(`   Error: ${error.message}`);
    }

    // Example 5: WebSocket events (optional)
    console.log('\n5. WebSocket Real-Time Events');
    try {
        // Uncomment to test WebSocket
        /*
        const ws = client.connectWebSocket(
            (message) => {
                console.log('   WebSocket message:', message);
            },
            (error) => {
                console.error('   WebSocket error:', error);
            }
        );

        // Subscribe to block events
        client.subscribe(ws, 'blocks');

        // Keep alive for a few seconds
        setTimeout(() => {
            ws.close();
        }, 5000);
        */
        console.log('   (Uncomment code to test WebSocket)');
    } catch (error) {
        console.error(`   Error: ${error.message}`);
    }

    console.log('\n=== Example Complete ===');
}

// Run if executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = IPFRSClient;
