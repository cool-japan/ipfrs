//! S3-compatible block storage backend
//!
//! Supports AWS S3, MinIO, Cloudflare R2, and other S3-compatible object stores.
//! Provides automatic multipart uploads for large blocks and efficient batch operations.

use crate::traits::BlockStore;
use async_trait::async_trait;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{Delete, ObjectIdentifier};
use aws_sdk_s3::Client as S3Client;
use futures::future::join_all;
use ipfrs_core::{Block, Cid, Error, Result};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// S3 block store configuration
#[derive(Debug, Clone)]
pub struct S3Config {
    /// S3 bucket name
    pub bucket: String,
    /// Optional prefix for all keys (e.g., "ipfrs/blocks/")
    pub prefix: Option<String>,
    /// Region (for AWS S3)
    pub region: Option<String>,
    /// Custom endpoint (for MinIO, R2, etc.)
    pub endpoint: Option<String>,
    /// Multipart upload threshold in bytes (default: 5MB)
    pub multipart_threshold: usize,
    /// Maximum concurrent operations (default: 10)
    pub max_concurrent: usize,
}

impl S3Config {
    /// Create a new S3 configuration
    pub fn new(bucket: String) -> Self {
        Self {
            bucket,
            prefix: None,
            region: None,
            endpoint: None,
            multipart_threshold: 5 * 1024 * 1024, // 5MB
            max_concurrent: 10,
        }
    }

    /// Set key prefix
    pub fn with_prefix(mut self, prefix: String) -> Self {
        self.prefix = Some(prefix);
        self
    }

    /// Set region (for AWS S3)
    pub fn with_region(mut self, region: String) -> Self {
        self.region = Some(region);
        self
    }

    /// Set custom endpoint (for MinIO, R2, etc.)
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    /// Set multipart upload threshold
    pub fn with_multipart_threshold(mut self, threshold: usize) -> Self {
        self.multipart_threshold = threshold;
        self
    }

    /// Set maximum concurrent operations
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Build the full S3 key from a CID
    fn build_key(&self, cid: &Cid) -> String {
        let cid_str = cid.to_string();
        match &self.prefix {
            Some(prefix) => format!("{prefix}{cid_str}"),
            None => cid_str,
        }
    }
}

/// Block storage using S3-compatible object store
#[derive(Clone)]
pub struct S3BlockStore {
    client: Arc<S3Client>,
    config: S3Config,
}

impl S3BlockStore {
    /// Create a new S3 block store
    pub async fn new(config: S3Config) -> Result<Self> {
        let aws_config = if let Some(endpoint) = &config.endpoint {
            // Custom endpoint (MinIO, R2, etc.)
            let mut builder = aws_config::defaults(aws_config::BehaviorVersion::latest());

            if let Some(region) = &config.region {
                builder = builder.region(aws_sdk_s3::config::Region::new(region.clone()));
            }

            let aws_config = builder.load().await;

            aws_sdk_s3::config::Builder::from(&aws_config)
                .endpoint_url(endpoint)
                .force_path_style(true) // Required for MinIO
                .build()
        } else {
            // AWS S3
            let mut builder = aws_config::defaults(aws_config::BehaviorVersion::latest());

            if let Some(region) = &config.region {
                builder = builder.region(aws_sdk_s3::config::Region::new(region.clone()));
            }

            let aws_config = builder.load().await;
            aws_sdk_s3::config::Builder::from(&aws_config).build()
        };

        let client = S3Client::from_conf(aws_config);

        Ok(Self {
            client: Arc::new(client),
            config,
        })
    }

    /// Get reference to S3 client
    pub fn client(&self) -> &Arc<S3Client> {
        &self.client
    }

    /// Get configuration
    pub fn config(&self) -> &S3Config {
        &self.config
    }

    /// Simple put for small blocks
    async fn put_simple(&self, key: &str, data: &[u8]) -> Result<()> {
        self.client
            .put_object()
            .bucket(&self.config.bucket)
            .key(key)
            .body(ByteStream::from(data.to_vec()))
            .send()
            .await
            .map_err(|e| Error::Storage(format!("Failed to put block to S3: {e}")))?;

        Ok(())
    }

    /// Multipart upload for large blocks with optimizations
    async fn put_multipart(&self, key: &str, data: &[u8]) -> Result<()> {
        // Dynamic part size calculation
        // For very large files (>1GB), use larger parts to reduce API calls
        let base_part_size = 5 * 1024 * 1024; // 5MB minimum
        let data_size = data.len();
        let part_size = if data_size > 1024 * 1024 * 1024 {
            // For files >1GB, use 10MB parts
            std::cmp::max(10 * 1024 * 1024, self.config.multipart_threshold)
        } else if data_size > 100 * 1024 * 1024 {
            // For files >100MB, use 8MB parts
            std::cmp::max(8 * 1024 * 1024, self.config.multipart_threshold)
        } else {
            // For smaller files, use 5MB parts
            std::cmp::max(base_part_size, self.config.multipart_threshold)
        };

        // Initiate multipart upload
        let multipart = self
            .client
            .create_multipart_upload()
            .bucket(&self.config.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| Error::Storage(format!("Failed to initiate multipart upload: {e}")))?;

        let upload_id = multipart
            .upload_id()
            .ok_or_else(|| Error::Storage("No upload ID returned".to_string()))?;

        // Split data into parts and upload concurrently
        let chunks: Vec<_> = data.chunks(part_size).collect();
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent));

        let mut futures = Vec::new();

        for (part_number, chunk) in chunks.iter().enumerate() {
            let part_num = (part_number + 1) as i32;
            let data_chunk = chunk.to_vec();
            let client = self.client.clone();
            let bucket = self.config.bucket.clone();
            let key = key.to_string();
            let upload_id = upload_id.to_string();
            let sem = semaphore.clone();

            let future = async move {
                let _permit = sem
                    .acquire()
                    .await
                    .expect("semaphore is never explicitly closed");

                // Retry logic for failed uploads (up to 3 attempts)
                let mut attempts = 0;
                let max_attempts = 3;
                let mut last_error = None;

                while attempts < max_attempts {
                    match client
                        .upload_part()
                        .bucket(&bucket)
                        .key(&key)
                        .upload_id(&upload_id)
                        .part_number(part_num)
                        .body(ByteStream::from(data_chunk.clone()))
                        .send()
                        .await
                    {
                        Ok(output) => {
                            return Ok((part_num, output.e_tag().unwrap_or_default().to_string()));
                        }
                        Err(e) => {
                            attempts += 1;
                            last_error = Some(e);
                            if attempts < max_attempts {
                                // Exponential backoff: 100ms, 200ms, 400ms
                                tokio::time::sleep(tokio::time::Duration::from_millis(
                                    100 * (1 << (attempts - 1)),
                                ))
                                .await;
                            }
                        }
                    }
                }

                Err(last_error.expect("loop ran at least max_attempts times so last_error is set"))
            };

            futures.push(future);
        }

        // Execute all uploads concurrently
        let results = join_all(futures).await;

        // Collect completed parts and sort by part number
        let mut upload_parts = Vec::new();
        for result in results {
            match result {
                Ok((part_number, etag)) => {
                    upload_parts.push((
                        part_number,
                        aws_sdk_s3::types::CompletedPart::builder()
                            .part_number(part_number)
                            .e_tag(etag)
                            .build(),
                    ));
                }
                Err(e) => {
                    // Abort the multipart upload on error
                    let _ = self
                        .client
                        .abort_multipart_upload()
                        .bucket(&self.config.bucket)
                        .key(key)
                        .upload_id(upload_id)
                        .send()
                        .await;

                    return Err(Error::Storage(format!(
                        "Failed to upload part after retries: {e}"
                    )));
                }
            }
        }

        // Sort parts by part number (required by S3)
        upload_parts.sort_by_key(|(part_num, _)| *part_num);
        let sorted_parts: Vec<_> = upload_parts.into_iter().map(|(_, part)| part).collect();

        // Complete the multipart upload
        let completed_upload = aws_sdk_s3::types::CompletedMultipartUpload::builder()
            .set_parts(Some(sorted_parts))
            .build();

        self.client
            .complete_multipart_upload()
            .bucket(&self.config.bucket)
            .key(key)
            .upload_id(upload_id)
            .multipart_upload(completed_upload)
            .send()
            .await
            .map_err(|e| Error::Storage(format!("Failed to complete multipart upload: {e}")))?;

        Ok(())
    }
}

#[async_trait]
impl BlockStore for S3BlockStore {
    /// Store a block
    async fn put(&self, block: &Block) -> Result<()> {
        let key = self.config.build_key(block.cid());
        let data = block.data();

        // Use multipart upload for large blocks
        if data.len() >= self.config.multipart_threshold {
            self.put_multipart(&key, data).await
        } else {
            self.put_simple(&key, data).await
        }
    }

    /// Retrieve a block by CID
    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        let key = self.config.build_key(cid);

        match self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await
        {
            Ok(output) => {
                let data = output
                    .body
                    .collect()
                    .await
                    .map_err(|e| Error::Storage(format!("Failed to read S3 object body: {e}")))?
                    .into_bytes();

                Ok(Some(Block::from_parts(*cid, data)))
            }
            Err(e) => {
                // Check if it's a "not found" error
                let error_str = e.to_string();
                if error_str.contains("NoSuchKey") || error_str.contains("404") {
                    Ok(None)
                } else {
                    Err(Error::Storage(format!("Failed to get block from S3: {e}")))
                }
            }
        }
    }

    /// Check if a block exists
    async fn has(&self, cid: &Cid) -> Result<bool> {
        let key = self.config.build_key(cid);

        match self
            .client
            .head_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => {
                let error_str = e.to_string();
                if error_str.contains("NotFound") || error_str.contains("404") {
                    Ok(false)
                } else {
                    Err(Error::Storage(format!(
                        "Failed to check block existence in S3: {e}"
                    )))
                }
            }
        }
    }

    /// Delete a block
    async fn delete(&self, cid: &Cid) -> Result<()> {
        let key = self.config.build_key(cid);

        self.client
            .delete_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| Error::Storage(format!("Failed to delete block from S3: {e}")))?;

        Ok(())
    }

    /// Get the number of blocks stored
    fn len(&self) -> usize {
        // S3 doesn't provide an efficient way to count objects
        // Return 0 and users should track this separately if needed
        0
    }

    /// Check if the store is empty
    fn is_empty(&self) -> bool {
        // Since len() is not efficient, we can't reliably check emptiness
        false
    }

    /// Get all CIDs in the store
    fn list_cids(&self) -> Result<Vec<Cid>> {
        // S3 listing is async, so we can't implement this in a sync function
        // Users should use a separate async method for listing
        // Return empty vec as a placeholder
        Ok(Vec::new())
    }

    /// Store multiple blocks (parallel with rate limiting)
    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        if blocks.is_empty() {
            return Ok(());
        }

        // Use semaphore to limit concurrent uploads
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent));
        let mut tasks = Vec::with_capacity(blocks.len());

        for block in blocks {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| Error::Storage(format!("Failed to acquire semaphore: {e}")))?;
            let block = block.clone();
            let store = self.clone();

            tasks.push(tokio::spawn(async move {
                let result = store.put(&block).await;
                drop(permit); // Release the permit
                result
            }));
        }

        // Wait for all uploads to complete
        let results = join_all(tasks).await;

        // Check for errors
        for result in results {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(Error::Storage(format!("Task join error: {e}"))),
            }
        }

        Ok(())
    }

    /// Retrieve multiple blocks (parallel with rate limiting)
    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        if cids.is_empty() {
            return Ok(Vec::new());
        }

        // Use semaphore to limit concurrent downloads
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent));
        let mut tasks = Vec::with_capacity(cids.len());

        for cid in cids {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| Error::Storage(format!("Failed to acquire semaphore: {e}")))?;
            let cid = *cid;
            let store = self.clone();

            tasks.push(tokio::spawn(async move {
                let result = store.get(&cid).await;
                drop(permit); // Release the permit
                result
            }));
        }

        // Wait for all downloads to complete
        let results = join_all(tasks).await;

        // Collect results in order
        let mut blocks = Vec::with_capacity(cids.len());
        for result in results {
            match result {
                Ok(Ok(block)) => blocks.push(block),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(Error::Storage(format!("Task join error: {e}"))),
            }
        }

        Ok(blocks)
    }

    /// Check if multiple blocks exist (parallel with rate limiting)
    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        if cids.is_empty() {
            return Ok(Vec::new());
        }

        // Use semaphore to limit concurrent checks
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent));
        let mut tasks = Vec::with_capacity(cids.len());

        for cid in cids {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| Error::Storage(format!("Failed to acquire semaphore: {e}")))?;
            let cid = *cid;
            let store = self.clone();

            tasks.push(tokio::spawn(async move {
                let result = store.has(&cid).await;
                drop(permit); // Release the permit
                result
            }));
        }

        // Wait for all checks to complete
        let results = join_all(tasks).await;

        // Collect results in order
        let mut exists_vec = Vec::with_capacity(cids.len());
        for result in results {
            match result {
                Ok(Ok(exists)) => exists_vec.push(exists),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(Error::Storage(format!("Task join error: {e}"))),
            }
        }

        Ok(exists_vec)
    }

    /// Delete multiple blocks using batch delete API
    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        if cids.is_empty() {
            return Ok(());
        }

        // S3 supports batch delete (up to 1000 objects per request)
        const BATCH_SIZE: usize = 1000;

        for chunk in cids.chunks(BATCH_SIZE) {
            // Build list of object identifiers
            let mut objects = Vec::with_capacity(chunk.len());
            for cid in chunk {
                let key = self.config.build_key(cid);
                objects.push(ObjectIdentifier::builder().key(key).build().map_err(|e| {
                    Error::Storage(format!("Failed to build object identifier: {e}"))
                })?);
            }

            // Create delete request
            let delete = Delete::builder()
                .set_objects(Some(objects))
                .build()
                .map_err(|e| Error::Storage(format!("Failed to build delete request: {e}")))?;

            // Execute batch delete
            self.client
                .delete_objects()
                .bucket(&self.config.bucket)
                .delete(delete)
                .send()
                .await
                .map_err(|e| Error::Storage(format!("Failed to delete objects: {e}")))?;
        }

        Ok(())
    }

    /// Flush is a no-op for S3 (writes are immediate)
    async fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// Helper methods for S3BlockStore
impl S3BlockStore {
    /// List all CIDs in the store (async version)
    pub async fn list_cids_async(&self) -> Result<Vec<Cid>> {
        let mut cids = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self.client.list_objects_v2().bucket(&self.config.bucket);

            if let Some(prefix) = &self.config.prefix {
                request = request.prefix(prefix);
            }

            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }

            let output = request
                .send()
                .await
                .map_err(|e| Error::Storage(format!("Failed to list S3 objects: {e}")))?;

            if let Some(contents) = output.contents {
                for object in contents {
                    if let Some(key) = object.key {
                        // Extract CID from key (remove prefix if present)
                        let cid_str = if let Some(prefix) = &self.config.prefix {
                            key.strip_prefix(prefix).unwrap_or(&key)
                        } else {
                            &key
                        };

                        // Parse CID
                        if let Ok(cid) = cid_str.parse::<Cid>() {
                            cids.push(cid);
                        }
                    }
                }
            }

            // Check if there are more results
            if output.is_truncated == Some(true) {
                continuation_token = output.next_continuation_token;
            } else {
                break;
            }
        }

        Ok(cids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_config_build_key() {
        let config = S3Config::new("test-bucket".to_string());
        let cid = "QmTest123".parse::<Cid>().unwrap_or_else(|_| {
            // Create a test CID
            use bytes::Bytes;
            *Block::new(Bytes::from("test")).unwrap().cid()
        });

        let key = config.build_key(&cid);
        assert_eq!(key, cid.to_string());

        // With prefix
        let config = config.with_prefix("ipfrs/blocks/".to_string());
        let key = config.build_key(&cid);
        assert_eq!(key, format!("ipfrs/blocks/{}", cid));
    }

    #[test]
    fn test_s3_config_builder() {
        let config = S3Config::new("test-bucket".to_string())
            .with_prefix("ipfrs/".to_string())
            .with_region("us-west-2".to_string())
            .with_endpoint("http://localhost:9000".to_string())
            .with_multipart_threshold(10 * 1024 * 1024);

        assert_eq!(config.bucket, "test-bucket");
        assert_eq!(config.prefix, Some("ipfrs/".to_string()));
        assert_eq!(config.region, Some("us-west-2".to_string()));
        assert_eq!(config.endpoint, Some("http://localhost:9000".to_string()));
        assert_eq!(config.multipart_threshold, 10 * 1024 * 1024);
    }
}
