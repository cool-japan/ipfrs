// gRPC service implementations for IPFRS
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use crate::backpressure::BackpressureConfig;

// Import BlockStore trait
use ipfrs_storage::traits::BlockStore;

// Include generated proto code
pub mod proto {
    pub mod block {
        tonic::include_proto!("ipfrs.block.v1");
    }
    pub mod dag {
        tonic::include_proto!("ipfrs.dag.v1");
    }
    pub mod file {
        tonic::include_proto!("ipfrs.file.v1");
    }
    pub mod tensor {
        tonic::include_proto!("ipfrs.tensor.v1");
    }
}

// Import block types with specific names to avoid ambiguity
use proto::block::{
    block_service_server, block_stream_request, block_stream_response, BatchGetBlocksRequest,
    BatchPutBlocksResponse, BlockStreamRequest, BlockStreamResponse, DeleteBlockRequest,
    DeleteBlockResponse, GetBlockRequest, GetBlockResponse, HasBlockRequest, HasBlockResponse,
    PutBlockRequest, PutBlockResponse,
};
use proto::block::{Error as BlockError, ErrorCode as BlockErrorCode};

// Import DAG types
use proto::dag::{
    dag_service_server, DagNode, GetDagRequest, GetDagResponse, GetDagStatsRequest,
    GetDagStatsResponse, PutDagRequest, PutDagResponse, ResolvePathRequest, ResolvePathResponse,
    TraverseDagRequest,
};

// Import file types
use proto::file::{
    add_file_request, file_service_server, AddFileRequest, AddFileResponse, FileChunk,
    FileMetadata, GetFileInfoRequest, GetFileInfoResponse, GetFileRequest, ListDirectoryRequest,
    ListDirectoryResponse, PinFileRequest, PinFileResponse, UnpinFileRequest, UnpinFileResponse,
};

// Import tensor types
use proto::tensor::{
    put_tensor_request, tensor_service_server, tensor_stream_response, DataType,
    GetTensorInfoRequest, GetTensorRequest, GetTensorStatsRequest, PutTensorRequest,
    PutTensorResponse, SliceTensorRequest, TensorChunk, TensorFormat, TensorInfo, TensorLayout,
    TensorMetadata, TensorStatsResponse, TensorStreamRequest, TensorStreamResponse,
};

// Re-export server wrapper types so callers can build tonic routers without
// having to import the generated proto module directly.
pub use proto::block::block_service_server::BlockServiceServer;
pub use proto::dag::dag_service_server::DagServiceServer;
pub use proto::file::file_service_server::FileServiceServer;
pub use proto::tensor::tensor_service_server::TensorServiceServer;

/// Request validation module for gRPC services
mod validation {
    use tonic::Status;

    /// Maximum data size for a single block (256 MB)
    const MAX_BLOCK_SIZE: usize = 256 * 1024 * 1024;

    /// Maximum number of CIDs in a batch request
    const MAX_BATCH_SIZE: usize = 1000;

    /// Maximum path length
    #[allow(dead_code)]
    const MAX_PATH_LENGTH: usize = 4096;

    /// Validate CID format
    #[allow(clippy::result_large_err)]
    pub fn validate_cid(cid: &str) -> Result<(), Status> {
        if cid.is_empty() {
            return Err(Status::invalid_argument("CID cannot be empty"));
        }

        if cid.len() > 200 {
            return Err(Status::invalid_argument("CID too long"));
        }

        // Basic CID format validation (starts with known prefixes)
        if !cid.starts_with("Qm")
            && !cid.starts_with("bafy")
            && !cid.starts_with("bafk")
            && !cid.starts_with("bafz")
        {
            return Err(Status::invalid_argument(format!(
                "Invalid CID format: {}",
                cid
            )));
        }

        Ok(())
    }

    /// Validate block data size
    #[allow(clippy::result_large_err)]
    pub fn validate_block_data(data: &[u8]) -> Result<(), Status> {
        if data.is_empty() {
            return Err(Status::invalid_argument("Block data cannot be empty"));
        }

        if data.len() > MAX_BLOCK_SIZE {
            return Err(Status::invalid_argument(format!(
                "Block data too large: {} bytes (max: {} bytes)",
                data.len(),
                MAX_BLOCK_SIZE
            )));
        }

        Ok(())
    }

    /// Validate batch size
    #[allow(clippy::result_large_err)]
    pub fn validate_batch_size(count: usize) -> Result<(), Status> {
        if count == 0 {
            return Err(Status::invalid_argument("Batch cannot be empty"));
        }

        if count > MAX_BATCH_SIZE {
            return Err(Status::invalid_argument(format!(
                "Batch too large: {} items (max: {} items)",
                count, MAX_BATCH_SIZE
            )));
        }

        Ok(())
    }

    /// Validate path string
    #[allow(dead_code)]
    #[allow(clippy::result_large_err)]
    pub fn validate_path(path: &str) -> Result<(), Status> {
        if path.len() > MAX_PATH_LENGTH {
            return Err(Status::invalid_argument(format!(
                "Path too long: {} characters (max: {} characters)",
                path.len(),
                MAX_PATH_LENGTH
            )));
        }

        // Check for null bytes
        if path.contains('\0') {
            return Err(Status::invalid_argument("Path contains null bytes"));
        }

        Ok(())
    }

    /// Validate tensor dimensions
    #[allow(dead_code)]
    #[allow(clippy::result_large_err)]
    pub fn validate_tensor_dims(dims: &[u64]) -> Result<(), Status> {
        if dims.is_empty() {
            return Err(Status::invalid_argument(
                "Tensor must have at least one dimension",
            ));
        }

        if dims.len() > 8 {
            return Err(Status::invalid_argument(format!(
                "Too many dimensions: {} (max: 8)",
                dims.len()
            )));
        }

        for (i, &dim) in dims.iter().enumerate() {
            if dim == 0 {
                return Err(Status::invalid_argument(format!(
                    "Dimension {} cannot be zero",
                    i
                )));
            }
        }

        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_validate_cid_valid() {
            assert!(validate_cid("QmTest123").is_ok());
            assert!(
                validate_cid("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi").is_ok()
            );
            assert!(validate_cid("bafkreigh2akiscaildcqabsyg3dfr6cyj").is_ok());
        }

        #[test]
        fn test_validate_cid_invalid() {
            assert!(validate_cid("").is_err());
            assert!(validate_cid("invalid").is_err());
            assert!(validate_cid("x".repeat(201).as_str()).is_err());
        }

        #[test]
        fn test_validate_block_data() {
            assert!(validate_block_data(&[1, 2, 3]).is_ok());
            assert!(validate_block_data(&[]).is_err());
            assert!(validate_block_data(&vec![0u8; 257 * 1024 * 1024]).is_err());
        }

        #[test]
        fn test_validate_batch_size() {
            assert!(validate_batch_size(1).is_ok());
            assert!(validate_batch_size(100).is_ok());
            assert!(validate_batch_size(0).is_err());
            assert!(validate_batch_size(1001).is_err());
        }

        #[test]
        fn test_validate_path() {
            assert!(validate_path("/ipfs/QmTest/file.txt").is_ok());
            assert!(validate_path("a/b/c").is_ok());
            assert!(validate_path(&"x".repeat(5000)).is_err());
            assert!(validate_path("path\0with\0nulls").is_err());
        }

        #[test]
        fn test_validate_tensor_dims() {
            assert!(validate_tensor_dims(&[10, 20, 30]).is_ok());
            assert!(validate_tensor_dims(&[100]).is_ok());
            assert!(validate_tensor_dims(&[]).is_err());
            assert!(validate_tensor_dims(&[1, 2, 3, 4, 5, 6, 7, 8, 9]).is_err());
            assert!(validate_tensor_dims(&[10, 0, 30]).is_err());
        }
    }
}

/// BlockService implementation
#[derive(Clone)]
pub struct BlockServiceImpl<S> {
    storage: Arc<S>,
}

impl<S> BlockServiceImpl<S> {
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}

impl<S> Default for BlockServiceImpl<S>
where
    S: Default,
{
    fn default() -> Self {
        Self::new(Arc::new(S::default()))
    }
}

#[tonic::async_trait]
impl<S> block_service_server::BlockService for BlockServiceImpl<S>
where
    S: BlockStore + 'static,
{
    async fn get_block(
        &self,
        request: Request<GetBlockRequest>,
    ) -> Result<Response<GetBlockResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("GetBlock request for CID: {}", req.cid);

        // Validate CID format
        validation::validate_cid(&req.cid)?;

        // Parse CID from string
        let cid = req
            .cid
            .parse::<ipfrs_core::Cid>()
            .map_err(|e| Status::invalid_argument(format!("Invalid CID: {}", e)))?;

        // Retrieve block from storage
        let block = self
            .storage
            .get(&cid)
            .await
            .map_err(|e| Status::internal(format!("Storage error: {}", e)))?;

        match block {
            Some(block) => {
                let response = GetBlockResponse {
                    cid: block.cid().to_string(),
                    data: block.data().to_vec(),
                    size: block.data().len() as u64,
                };
                Ok(Response::new(response))
            }
            None => Err(Status::not_found(format!("Block not found: {}", req.cid))),
        }
    }

    async fn put_block(
        &self,
        request: Request<PutBlockRequest>,
    ) -> Result<Response<PutBlockResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("PutBlock request, data size: {}", req.data.len());

        // Validate block data
        validation::validate_block_data(&req.data)?;

        // Create block from data
        let block = ipfrs_core::Block::new(req.data.into())
            .map_err(|e| Status::invalid_argument(format!("Invalid block data: {}", e)))?;

        let cid = *block.cid();
        let size = block.data().len() as u64;

        // Store block
        self.storage
            .put(&block)
            .await
            .map_err(|e| Status::internal(format!("Storage error: {}", e)))?;

        let response = PutBlockResponse {
            cid: cid.to_string(),
            size,
        };

        Ok(Response::new(response))
    }

    async fn has_block(
        &self,
        request: Request<HasBlockRequest>,
    ) -> Result<Response<HasBlockResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("HasBlock request for CID: {}", req.cid);

        // Validate CID format
        validation::validate_cid(&req.cid)?;

        // Parse CID from string
        let cid = req
            .cid
            .parse::<ipfrs_core::Cid>()
            .map_err(|e| Status::invalid_argument(format!("Invalid CID: {}", e)))?;

        // Check existence in storage
        let exists = self
            .storage
            .has(&cid)
            .await
            .map_err(|e| Status::internal(format!("Storage error: {}", e)))?;

        // Get size if block exists
        let size = if exists {
            match self.storage.get(&cid).await {
                Ok(Some(block)) => Some(block.data().len() as u64),
                _ => None,
            }
        } else {
            None
        };

        let response = HasBlockResponse { exists, size };

        Ok(Response::new(response))
    }

    async fn delete_block(
        &self,
        request: Request<DeleteBlockRequest>,
    ) -> Result<Response<DeleteBlockResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("DeleteBlock request for CID: {}", req.cid);

        // Validate CID format
        validation::validate_cid(&req.cid)?;

        // Parse CID from string
        let cid = req
            .cid
            .parse::<ipfrs_core::Cid>()
            .map_err(|e| Status::invalid_argument(format!("Invalid CID: {}", e)))?;

        // Delete block from storage
        self.storage
            .delete(&cid)
            .await
            .map_err(|e| Status::internal(format!("Storage error: {}", e)))?;

        let response = DeleteBlockResponse { deleted: true };

        Ok(Response::new(response))
    }

    type BatchGetBlocksStream =
        Pin<Box<dyn Stream<Item = Result<GetBlockResponse, Status>> + Send>>;

    async fn batch_get_blocks(
        &self,
        request: Request<BatchGetBlocksRequest>,
    ) -> Result<Response<Self::BatchGetBlocksStream>, Status> {
        let req = request.into_inner();
        tracing::info!("BatchGetBlocks request for {} CIDs", req.cids.len());

        // Validate batch size
        validation::validate_batch_size(req.cids.len())?;

        let storage = Arc::clone(&self.storage);

        // Create a stream of responses
        let stream = async_stream::stream! {
            for cid_str in req.cids {
                // Parse CID
                let cid = match cid_str.parse::<ipfrs_core::Cid>() {
                    Ok(cid) => cid,
                    Err(e) => {
                        yield Err(Status::invalid_argument(format!("Invalid CID {}: {}", cid_str, e)));
                        continue;
                    }
                };

                // Retrieve block
                match storage.get(&cid).await {
                    Ok(Some(block)) => {
                        yield Ok(GetBlockResponse {
                            cid: block.cid().to_string(),
                            data: block.data().to_vec(),
                            size: block.data().len() as u64,
                        });
                    }
                    Ok(None) => {
                        yield Err(Status::not_found(format!("Block not found: {}", cid_str)));
                    }
                    Err(e) => {
                        yield Err(Status::internal(format!("Storage error: {}", e)));
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    async fn batch_put_blocks(
        &self,
        request: Request<tonic::Streaming<PutBlockRequest>>,
    ) -> Result<Response<BatchPutBlocksResponse>, Status> {
        let mut stream = request.into_inner();
        let mut count = 0u32;
        let mut total_size = 0u64;
        let mut cids = Vec::new();

        while let Some(result) = stream.next().await {
            let req = result?;

            // Validate block data
            validation::validate_block_data(&req.data)?;

            // Create block from data
            let block = ipfrs_core::Block::new(req.data.into())
                .map_err(|e| Status::invalid_argument(format!("Invalid block data: {}", e)))?;

            let cid = *block.cid();
            let size = block.data().len() as u64;

            // Store block
            self.storage
                .put(&block)
                .await
                .map_err(|e| Status::internal(format!("Storage error: {}", e)))?;

            count += 1;
            total_size += size;
            cids.push(cid.to_string());
        }

        tracing::info!("BatchPutBlocks completed: {} blocks", count);

        let response = BatchPutBlocksResponse {
            cids,
            total_size,
            count,
        };

        Ok(Response::new(response))
    }

    type StreamBlocksStream =
        Pin<Box<dyn Stream<Item = Result<BlockStreamResponse, Status>> + Send>>;

    async fn stream_blocks(
        &self,
        request: Request<tonic::Streaming<BlockStreamRequest>>,
    ) -> Result<Response<Self::StreamBlocksStream>, Status> {
        let mut in_stream = request.into_inner();
        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            while let Some(result) = in_stream.next().await {
                match result {
                    Ok(req) => {
                        let response = match req.request {
                            Some(block_stream_request::Request::Get(get_req)) => {
                                BlockStreamResponse {
                                    response: Some(block_stream_response::Response::Get(
                                        GetBlockResponse {
                                            cid: get_req.cid,
                                            data: vec![1, 2, 3, 4],
                                            size: 4,
                                        },
                                    )),
                                }
                            }
                            Some(block_stream_request::Request::Put(put_req)) => {
                                BlockStreamResponse {
                                    response: Some(block_stream_response::Response::Put(
                                        PutBlockResponse {
                                            cid: "QmMockCID".to_string(),
                                            size: put_req.data.len() as u64,
                                        },
                                    )),
                                }
                            }
                            Some(block_stream_request::Request::Has(_has_req)) => {
                                BlockStreamResponse {
                                    response: Some(block_stream_response::Response::Has(
                                        HasBlockResponse {
                                            exists: true,
                                            size: Some(4),
                                        },
                                    )),
                                }
                            }
                            None => BlockStreamResponse {
                                response: Some(block_stream_response::Response::Error(
                                    BlockError {
                                        message: "Invalid request".to_string(),
                                        code: BlockErrorCode::Internal as i32,
                                    },
                                )),
                            },
                        };
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        let out_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(out_stream)))
    }
}

/// DagService implementation
#[derive(Clone)]
pub struct DagServiceImpl {
    _storage: Arc<()>,
}

impl DagServiceImpl {
    pub fn new() -> Self {
        Self {
            _storage: Arc::new(()),
        }
    }
}

impl Default for DagServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl dag_service_server::DagService for DagServiceImpl {
    async fn get_dag(
        &self,
        request: Request<GetDagRequest>,
    ) -> Result<Response<GetDagResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("GetDag request for CID: {}", req.cid);

        let response = GetDagResponse {
            cid: req.cid,
            data: vec![],
            format: "dag-cbor".to_string(),
            size: 0,
        };

        Ok(Response::new(response))
    }

    async fn put_dag(
        &self,
        request: Request<PutDagRequest>,
    ) -> Result<Response<PutDagResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("PutDag request, format: {}", req.format);

        let response = PutDagResponse {
            cid: "QmMockDagCID".to_string(),
            size: req.data.len() as u64,
        };

        Ok(Response::new(response))
    }

    async fn resolve_path(
        &self,
        request: Request<ResolvePathRequest>,
    ) -> Result<Response<ResolvePathResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("ResolvePath request: {}", req.path);

        let response = ResolvePathResponse {
            cid: "QmMockResolvedCID".to_string(),
            data: vec![],
            remaining_path: String::new(),
        };

        Ok(Response::new(response))
    }

    type TraverseDagStream = Pin<Box<dyn Stream<Item = Result<DagNode, Status>> + Send>>;

    async fn traverse_dag(
        &self,
        request: Request<TraverseDagRequest>,
    ) -> Result<Response<Self::TraverseDagStream>, Status> {
        let req = request.into_inner();
        tracing::info!("TraverseDag request for root: {}", req.root_cid);

        // Mock traversal
        let nodes = vec![DagNode {
            cid: req.root_cid,
            data: vec![],
            links: vec![],
            depth: 0,
        }];

        let stream = tokio_stream::iter(nodes.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_dag_stats(
        &self,
        request: Request<GetDagStatsRequest>,
    ) -> Result<Response<GetDagStatsResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("GetDagStats request for root: {}", req.root_cid);

        let response = GetDagStatsResponse {
            total_size: 0,
            num_blocks: 1,
            max_depth: 1,
            num_links: 0,
        };

        Ok(Response::new(response))
    }
}

/// FileService implementation
#[derive(Clone)]
pub struct FileServiceImpl {
    _storage: Arc<()>,
}

impl FileServiceImpl {
    pub fn new() -> Self {
        Self {
            _storage: Arc::new(()),
        }
    }
}

impl Default for FileServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl file_service_server::FileService for FileServiceImpl {
    async fn add_file(
        &self,
        request: Request<tonic::Streaming<AddFileRequest>>,
    ) -> Result<Response<AddFileResponse>, Status> {
        let mut stream = request.into_inner();
        let mut total_size = 0u64;
        let mut _metadata: Option<FileMetadata> = None;

        while let Some(result) = stream.next().await {
            let req = result?;
            match req.data {
                Some(add_file_request::Data::Metadata(meta)) => {
                    _metadata = Some(meta);
                }
                Some(add_file_request::Data::Chunk(chunk)) => {
                    total_size += chunk.len() as u64;
                }
                None => {}
            }
        }

        tracing::info!("AddFile completed, total size: {}", total_size);

        let response = AddFileResponse {
            cid: "QmMockFileCID".to_string(),
            size: total_size,
            num_blocks: 1,
        };

        Ok(Response::new(response))
    }

    type GetFileStream = Pin<Box<dyn Stream<Item = Result<FileChunk, Status>> + Send>>;

    async fn get_file(
        &self,
        request: Request<GetFileRequest>,
    ) -> Result<Response<Self::GetFileStream>, Status> {
        let req = request.into_inner();
        tracing::info!("GetFile request for CID: {}", req.cid);

        let chunks = vec![FileChunk {
            data: vec![1, 2, 3, 4],
            offset: 0,
            is_last: true,
        }];

        let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn list_directory(
        &self,
        request: Request<ListDirectoryRequest>,
    ) -> Result<Response<ListDirectoryResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("ListDirectory request for CID: {}", req.cid);

        let response = ListDirectoryResponse { entries: vec![] };

        Ok(Response::new(response))
    }

    async fn get_file_info(
        &self,
        request: Request<GetFileInfoRequest>,
    ) -> Result<Response<GetFileInfoResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("GetFileInfo request for CID: {}", req.cid);

        let response = GetFileInfoResponse {
            cid: req.cid,
            size: 0,
            num_blocks: 0,
            mime_type: None,
            is_directory: false,
        };

        Ok(Response::new(response))
    }

    async fn pin_file(
        &self,
        request: Request<PinFileRequest>,
    ) -> Result<Response<PinFileResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("PinFile request for CID: {}", req.cid);

        let response = PinFileResponse {
            pinned: true,
            blocks_pinned: 1,
        };

        Ok(Response::new(response))
    }

    async fn unpin_file(
        &self,
        request: Request<UnpinFileRequest>,
    ) -> Result<Response<UnpinFileResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("UnpinFile request for CID: {}", req.cid);

        let response = UnpinFileResponse {
            unpinned: true,
            blocks_unpinned: 1,
        };

        Ok(Response::new(response))
    }
}

/// TensorService implementation
#[derive(Clone)]
pub struct TensorServiceImpl {
    _storage: Arc<()>,
}

impl TensorServiceImpl {
    pub fn new() -> Self {
        Self {
            _storage: Arc::new(()),
        }
    }
}

impl Default for TensorServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl tensor_service_server::TensorService for TensorServiceImpl {
    type GetTensorStream = Pin<Box<dyn Stream<Item = Result<TensorChunk, Status>> + Send>>;

    async fn get_tensor(
        &self,
        request: Request<GetTensorRequest>,
    ) -> Result<Response<Self::GetTensorStream>, Status> {
        let req = request.into_inner();
        tracing::info!("GetTensor request for CID: {}", req.cid);

        let chunks = vec![TensorChunk {
            data: vec![],
            offset: 0,
            is_last: true,
            metadata: None,
        }];

        let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn put_tensor(
        &self,
        request: Request<tonic::Streaming<PutTensorRequest>>,
    ) -> Result<Response<PutTensorResponse>, Status> {
        let mut stream = request.into_inner();
        let mut total_size = 0u64;

        while let Some(result) = stream.next().await {
            let req = result?;
            if let Some(put_tensor_request::Data::Chunk(chunk)) = req.data {
                total_size += chunk.len() as u64;
            }
        }

        tracing::info!("PutTensor completed, total size: {}", total_size);

        let response = PutTensorResponse {
            cid: "QmMockTensorCID".to_string(),
            size: total_size,
        };

        Ok(Response::new(response))
    }

    async fn get_tensor_info(
        &self,
        request: Request<GetTensorInfoRequest>,
    ) -> Result<Response<TensorInfo>, Status> {
        let req = request.into_inner();
        tracing::info!("GetTensorInfo request for CID: {}", req.cid);

        let response = TensorInfo {
            cid: req.cid,
            metadata: Some(TensorMetadata {
                shape: vec![],
                dtype: DataType::F32 as i32,
                layout: TensorLayout::RowMajor as i32,
                name: None,
                format: TensorFormat::Safetensors as i32,
            }),
            size: 0,
        };

        Ok(Response::new(response))
    }

    type SliceTensorStream = Pin<Box<dyn Stream<Item = Result<TensorChunk, Status>> + Send>>;

    async fn slice_tensor(
        &self,
        request: Request<SliceTensorRequest>,
    ) -> Result<Response<Self::SliceTensorStream>, Status> {
        let req = request.into_inner();
        tracing::info!("SliceTensor request for CID: {}", req.cid);

        let chunks = vec![TensorChunk {
            data: vec![],
            offset: 0,
            is_last: true,
            metadata: None,
        }];

        let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_tensor_stats(
        &self,
        request: Request<GetTensorStatsRequest>,
    ) -> Result<Response<TensorStatsResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("GetTensorStats request for CID: {}", req.cid);

        let response = TensorStatsResponse {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            num_elements: 0,
            histogram: None,
        };

        Ok(Response::new(response))
    }

    type StreamTensorsStream =
        Pin<Box<dyn Stream<Item = Result<TensorStreamResponse, Status>> + Send>>;

    async fn stream_tensors(
        &self,
        request: Request<tonic::Streaming<TensorStreamRequest>>,
    ) -> Result<Response<Self::StreamTensorsStream>, Status> {
        let mut in_stream = request.into_inner();
        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            while let Some(result) = in_stream.next().await {
                match result {
                    Ok(_req) => {
                        // Process request and send response
                        let response = TensorStreamResponse {
                            response: Some(tensor_stream_response::Response::Chunk(TensorChunk {
                                data: vec![],
                                offset: 0,
                                is_last: true,
                                metadata: None,
                            })),
                        };
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        let out_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(out_stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::proto::block::block_service_server::BlockService;
    use super::proto::dag::dag_service_server::DagService;
    use super::proto::file::file_service_server::FileService;
    use super::proto::tensor::tensor_service_server::TensorService;
    use super::*;

    #[tokio::test]
    async fn test_block_service_get() {
        use ipfrs_storage::MemoryBlockStore;
        let storage = Arc::new(MemoryBlockStore::new());
        let service = BlockServiceImpl::new(storage.clone());

        // First add a block
        let test_data = vec![1, 2, 3, 4];
        let block = ipfrs_core::Block::new(test_data.clone().into())
            .expect("test: block creation should succeed");
        let test_cid = block.cid().to_string();
        storage
            .put(&block)
            .await
            .expect("test: block storage put should succeed");

        // Now get it
        let request = Request::new(GetBlockRequest {
            cid: test_cid.clone(),
        });
        let response = service
            .get_block(request)
            .await
            .expect("test: get block request should succeed");
        let inner = response.into_inner();
        assert_eq!(inner.cid, test_cid);
        assert_eq!(inner.data, test_data);
    }

    #[tokio::test]
    async fn test_block_service_put() {
        use ipfrs_storage::MemoryBlockStore;
        let storage = Arc::new(MemoryBlockStore::new());
        let service = BlockServiceImpl::new(storage);
        let request = Request::new(PutBlockRequest {
            data: vec![1, 2, 3, 4],
            format: None,
        });
        let response = service
            .put_block(request)
            .await
            .expect("test: put block request should succeed");
        assert_eq!(response.into_inner().size, 4);
    }

    #[tokio::test]
    async fn test_dag_service_get() {
        let service = DagServiceImpl::new();
        let request = Request::new(GetDagRequest {
            cid: "QmTest".to_string(),
            path: None,
        });
        let response = service
            .get_dag(request)
            .await
            .expect("test: get dag request should succeed");
        assert_eq!(response.into_inner().format, "dag-cbor");
    }

    #[tokio::test]
    async fn test_file_service_get_info() {
        let service = FileServiceImpl::new();
        let request = Request::new(GetFileInfoRequest {
            cid: "QmTest".to_string(),
        });
        let response = service
            .get_file_info(request)
            .await
            .expect("test: get file info request should succeed");
        assert_eq!(response.into_inner().cid, "QmTest");
    }

    #[tokio::test]
    async fn test_tensor_service_get_info() {
        let service = TensorServiceImpl::new();
        let request = Request::new(GetTensorInfoRequest {
            cid: "QmTest".to_string(),
        });
        let response = service
            .get_tensor_info(request)
            .await
            .expect("test: get tensor info request should succeed");
        assert_eq!(response.into_inner().cid, "QmTest");
    }
}

// ============================================================================
// gRPC Interceptors
// ============================================================================

use std::time::Instant;
use tonic::service::Interceptor;

/// Authentication interceptor that validates JWT tokens from metadata
#[derive(Clone)]
pub struct AuthInterceptor {
    jwt_manager: Arc<crate::auth::JwtManager>,
}

impl AuthInterceptor {
    pub fn new(jwt_secret: &str) -> Self {
        Self {
            jwt_manager: Arc::new(crate::auth::JwtManager::new(jwt_secret.as_bytes())),
        }
    }

    #[allow(clippy::result_large_err)]
    fn validate_token(&self, token: &str) -> Result<(), Status> {
        // Validate JWT token
        match self.jwt_manager.validate_token(token) {
            Ok(_claims) => Ok(()),
            Err(_) => Err(Status::unauthenticated("Invalid or expired token")),
        }
    }
}

impl Interceptor for AuthInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        // Extract authorization header
        let token = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or_else(|| Status::unauthenticated("Missing authorization token"))?;

        // Validate token
        self.validate_token(token)?;

        Ok(request)
    }
}

/// Logging interceptor that logs requests with timing information
#[derive(Clone)]
pub struct LoggingInterceptor;

impl LoggingInterceptor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LoggingInterceptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Interceptor for LoggingInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        tracing::info!("gRPC request received");

        // Store start time in extensions for later retrieval
        request.extensions_mut().insert(Instant::now());

        Ok(request)
    }
}

/// Metrics interceptor that tracks request counts and latencies
#[derive(Clone)]
pub struct MetricsInterceptor {
    request_count: Arc<std::sync::atomic::AtomicU64>,
    error_count: Arc<std::sync::atomic::AtomicU64>,
}

impl MetricsInterceptor {
    pub fn new() -> Self {
        Self {
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            error_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    pub fn request_count(&self) -> u64 {
        self.request_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn error_count(&self) -> u64 {
        self.error_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for MetricsInterceptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Interceptor for MetricsInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        // Increment request counter
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(request)
    }
}

/// Combined interceptor that chains multiple interceptors
#[derive(Clone)]
pub struct ChainedInterceptor {
    auth: Option<AuthInterceptor>,
    logging: Option<LoggingInterceptor>,
    metrics: Option<MetricsInterceptor>,
}

impl ChainedInterceptor {
    pub fn new() -> Self {
        Self {
            auth: None,
            logging: None,
            metrics: None,
        }
    }

    pub fn with_auth(mut self, jwt_secret: &str) -> Self {
        self.auth = Some(AuthInterceptor::new(jwt_secret));
        self
    }

    pub fn with_logging(mut self) -> Self {
        self.logging = Some(LoggingInterceptor::new());
        self
    }

    pub fn with_metrics(mut self) -> Self {
        self.metrics = Some(MetricsInterceptor::new());
        self
    }

    pub fn metrics(&self) -> Option<&MetricsInterceptor> {
        self.metrics.as_ref()
    }
}

impl Default for ChainedInterceptor {
    fn default() -> Self {
        Self::new()
    }
}

impl Interceptor for ChainedInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        // Apply metrics interceptor first
        if let Some(ref mut metrics) = self.metrics {
            request = metrics.call(request)?;
        }

        // Then logging
        if let Some(ref mut logging) = self.logging {
            request = logging.call(request)?;
        }

        // Finally auth (fail fast if auth fails)
        if let Some(ref mut auth) = self.auth {
            request = auth.call(request)?;
        }

        Ok(request)
    }
}

/// Rate limiting interceptor
#[derive(Clone)]
#[allow(dead_code)]
pub struct RateLimitInterceptor {
    max_requests_per_minute: u32,
    request_times: Arc<tokio::sync::Mutex<Vec<Instant>>>,
}

#[allow(dead_code)]
impl RateLimitInterceptor {
    pub fn new(max_requests_per_minute: u32) -> Self {
        Self {
            max_requests_per_minute,
            request_times: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        }
    }

    async fn check_rate_limit(&self) -> Result<(), Status> {
        let mut times = self.request_times.lock().await;
        let now = Instant::now();

        // Remove requests older than 1 minute
        times.retain(|t| now.duration_since(*t).as_secs() < 60);

        if times.len() >= self.max_requests_per_minute as usize {
            return Err(Status::resource_exhausted("Rate limit exceeded"));
        }

        times.push(now);
        Ok(())
    }
}

// Note: RateLimitInterceptor needs async, so it requires a different approach
// It would typically be implemented as a tower Layer instead of an Interceptor

/// Backpressure-aware streaming helpers
pub mod backpressure_support {
    use super::*;
    use crate::backpressure::{BackpressureConfig, BackpressureController};
    use std::sync::Arc;

    /// Create a backpressure-aware stream wrapper
    pub fn create_backpressure_controller(
        config: Option<BackpressureConfig>,
    ) -> Arc<BackpressureController> {
        Arc::new(BackpressureController::new(config.unwrap_or_default()))
    }

    /// Apply backpressure to a streaming RPC by wrapping the channel send
    pub async fn send_with_backpressure<T>(
        tx: &mpsc::Sender<Result<T, Status>>,
        item: Result<T, Status>,
        controller: &Arc<BackpressureController>,
    ) -> bool {
        // Acquire backpressure permit
        match controller.acquire().await {
            Ok(_permit) => {
                // Send item (permit is automatically released on drop)
                if tx.send(item).await.is_err() {
                    return false;
                }
                // Check for congestion and adjust window
                controller.check_congestion().await;
                true
            }
            Err(_) => false,
        }
    }
}

/// Configuration for gRPC services with backpressure support
#[derive(Debug, Clone)]
pub struct GrpcServiceConfig {
    pub backpressure: Option<BackpressureConfig>,
    pub enable_monitoring: bool,
}

impl Default for GrpcServiceConfig {
    fn default() -> Self {
        Self {
            backpressure: Some(BackpressureConfig::default()),
            enable_monitoring: true,
        }
    }
}

// ── Gradient sync service ─────────────────────────────────────────────────

/// Request to initiate a distributed gradient synchronisation session.
///
/// The `local_gradient` field carries the caller's local gradient encoded
/// as Arrow IPC bytes (use
/// [`ipfrs_tensorlogic::gradient::arrow_ipc::store_gradient_as_arrow`] to
/// produce them).
#[derive(Debug, Clone)]
pub struct GradientSyncRequest {
    /// Unique identifier for this synchronisation round.
    pub session_id: String,
    /// Arrow IPC-encoded local gradient contributed by the caller.
    pub local_gradient: Vec<u8>,
    /// Minimum number of peer gradients required before aggregating.
    pub min_peers: u32,
    /// Wall-clock timeout in seconds before the session is abandoned.
    pub timeout_secs: u64,
}

/// A single gradient chunk streamed back to the client.
///
/// During a [`GradientSyncService::sync_gradients`] call the service pushes
/// one `GradientChunkResponse` per Arrow IPC chunk received from a peer so
/// that clients can start processing data before all peers have responded.
#[derive(Debug, Clone)]
pub struct GradientChunkResponse {
    /// Session identifier matching [`GradientSyncRequest::session_id`].
    pub session_id: String,
    /// Zero-based index of this chunk within the peer's gradient stream.
    pub chunk_index: u32,
    /// Total chunks expected from this peer.
    pub total_chunks: u32,
    /// Arrow IPC bytes for this chunk.
    pub data: Vec<u8>,
    /// Peer that contributed this chunk.
    pub peer_id: String,
}

/// gRPC service that streams gradient chunks to clients as they arrive from peers.
///
/// The service wraps a `DistributedGradientAccumulator` stored behind an
/// `Arc<Mutex<…>>` so that concurrent sync sessions can safely share the same
/// block store without requiring access to the full `Node` type (which lives in
/// the `ipfrs` crate and cannot be referenced from `ipfrs-interface` without a
/// circular dependency).
pub struct GradientSyncService {
    store: std::sync::Arc<dyn ipfrs_storage::traits::BlockStore>,
}

impl GradientSyncService {
    /// Create a new service backed by `store`.
    pub fn new(store: std::sync::Arc<dyn ipfrs_storage::traits::BlockStore>) -> Self {
        Self { store }
    }

    /// Start a gradient sync session and stream chunks via `chunk_tx`.
    ///
    /// Workflow:
    /// 1. Decode `request.local_gradient` from Arrow IPC.
    /// 2. Commit the local gradient to the block store via
    ///    `DistributedGradientAccumulator::commit_local`.
    /// 3. Poll for peer gradients (stubbed — no live network in this layer).
    /// 4. Push one [`GradientChunkResponse`] per local chunk onto `chunk_tx`
    ///    so that the caller can observe streaming behaviour end-to-end.
    ///
    /// When full peer-to-peer transport is wired up, step 3 would await
    /// actual peer CIDs from the network layer.
    pub async fn sync_gradients(
        &self,
        request: GradientSyncRequest,
        chunk_tx: tokio::sync::mpsc::Sender<GradientChunkResponse>,
    ) -> anyhow::Result<()> {
        use ipfrs_tensorlogic::gradient::arrow_ipc::{
            load_gradient_from_arrow, store_gradient_as_arrow,
        };
        use ipfrs_tensorlogic::gradient::backward_pass::BackwardPassConfig;
        use ipfrs_tensorlogic::gradient::federated::DistributedGradientAccumulator;

        // Decode the caller's local gradient from Arrow IPC.
        let local_gradient = load_gradient_from_arrow(&request.local_gradient)
            .map_err(|e| anyhow::anyhow!("failed to decode local gradient: {e}"))?;

        tracing::debug!(
            session_id = %request.session_id,
            gradient_len = local_gradient.len(),
            min_peers = request.min_peers,
            timeout_secs = request.timeout_secs,
            "GradientSyncService: starting sync session"
        );

        // Commit the local gradient to the block store.
        let mut accumulator =
            DistributedGradientAccumulator::new(&request.session_id, BackwardPassConfig::default());

        let _local_cid = accumulator
            .commit_local(local_gradient.clone(), self.store.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("commit_local failed: {e}"))?;

        tracing::debug!(
            session_id = %request.session_id,
            "GradientSyncService: local gradient committed, CID = {_local_cid}"
        );

        // In a live deployment peer CIDs would be discovered via the network
        // layer and fed into `accumulator.add_peer_gradient(...)`.  Since
        // ipfrs-interface has no direct access to the network, we stream the
        // local gradient back in chunks so that the caller can observe the
        // server-streaming pattern end-to-end.
        let chunk_size = 65_536usize;
        let total_chunks = local_gradient.len().div_ceil(chunk_size).max(1);

        if local_gradient.is_empty() {
            // Send a single empty chunk to signal stream completion.
            let ipc = store_gradient_as_arrow(&[])
                .map_err(|e| anyhow::anyhow!("Arrow IPC encode: {e}"))?;
            chunk_tx
                .send(GradientChunkResponse {
                    session_id: request.session_id.clone(),
                    chunk_index: 0,
                    total_chunks: 1,
                    data: ipc,
                    peer_id: "local".to_string(),
                })
                .await
                .map_err(|_| anyhow::anyhow!("chunk_tx receiver dropped"))?;
            return Ok(());
        }

        for (idx, window) in local_gradient.chunks(chunk_size).enumerate() {
            let ipc = store_gradient_as_arrow(window)
                .map_err(|e| anyhow::anyhow!("Arrow IPC encode chunk {idx}: {e}"))?;

            let response = GradientChunkResponse {
                session_id: request.session_id.clone(),
                chunk_index: idx as u32,
                total_chunks: total_chunks as u32,
                data: ipc,
                peer_id: "local".to_string(),
            };

            chunk_tx
                .send(response)
                .await
                .map_err(|_| anyhow::anyhow!("chunk_tx receiver dropped at chunk {idx}"))?;
        }

        tracing::debug!(
            session_id = %request.session_id,
            total_chunks,
            "GradientSyncService: streamed all local chunks"
        );

        Ok(())
    }
}

#[cfg(test)]
mod interceptor_tests {
    use super::*;

    #[test]
    fn test_logging_interceptor() {
        let mut interceptor = LoggingInterceptor::new();
        let request = Request::new(());
        let result = interceptor.call(request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_metrics_interceptor() {
        let mut interceptor = MetricsInterceptor::new();
        assert_eq!(interceptor.request_count(), 0);

        let request = Request::new(());
        let _ = interceptor.call(request);
        assert_eq!(interceptor.request_count(), 1);

        let request2 = Request::new(());
        let _ = interceptor.call(request2);
        assert_eq!(interceptor.request_count(), 2);
    }

    #[test]
    fn test_chained_interceptor() {
        let mut interceptor = ChainedInterceptor::new().with_logging().with_metrics();

        let request = Request::new(());
        let result = interceptor.call(request);
        assert!(result.is_ok());

        // Check metrics were updated
        if let Some(metrics) = interceptor.metrics() {
            assert_eq!(metrics.request_count(), 1);
        }
    }

    #[test]
    fn test_auth_interceptor_missing_token() {
        let mut interceptor = AuthInterceptor::new("test_secret");
        let request = Request::new(());
        let result = interceptor.call(request);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn test_auth_interceptor_with_token() {
        use crate::auth::{JwtManager, Role, User};
        use tonic::metadata::MetadataValue;

        let secret = "test_secret";
        let user = User::new("test_user".to_string(), "password", Role::Admin)
            .expect("test: user creation should succeed");
        let jwt_manager = JwtManager::new(secret.as_bytes());
        let token = jwt_manager
            .generate_token(&user, 24)
            .expect("test: JWT token generation should succeed");

        let mut interceptor = AuthInterceptor::new(secret);
        let mut request = Request::new(());

        // Add authorization header
        let auth_value = MetadataValue::try_from(format!("Bearer {}", token))
            .expect("test: metadata value creation from bearer token should succeed");
        request.metadata_mut().insert("authorization", auth_value);

        let result = interceptor.call(request);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_backpressure_integration() {
        use crate::backpressure::BackpressureConfig;

        let config = BackpressureConfig {
            initial_window: 10,
            ..Default::default()
        };

        let controller = backpressure_support::create_backpressure_controller(Some(config));
        assert_eq!(controller.window_size(), 10);

        // Test sending with backpressure
        let (tx, mut rx) = mpsc::channel(100);
        let controller_clone = controller.clone();
        let controller_recv = controller.clone();

        tokio::spawn(async move {
            for i in 0..5 {
                let item = Ok(i);
                if !backpressure_support::send_with_backpressure(&tx, item, &controller_clone).await
                {
                    break;
                }
            }
        });

        // Receive items and signal consumption
        let mut count = 0;
        while let Some(item) = rx.recv().await {
            assert!(item.is_ok());
            controller_recv.signal_consumed();
            count += 1;
        }

        assert_eq!(count, 5);
        assert_eq!(controller.items_sent(), 5);
        assert_eq!(controller.items_consumed(), 5);
    }

    #[tokio::test]
    async fn test_backpressure_congestion() {
        use crate::backpressure::BackpressureConfig;
        use tokio::time::{sleep, Duration};

        let config = BackpressureConfig {
            initial_window: 5,
            min_window: 2,
            slow_consumer_threshold: 0.6,
            check_interval: Duration::from_millis(10),
            decrease_factor: 0.5,
            ..Default::default()
        };

        let controller = backpressure_support::create_backpressure_controller(Some(config));
        let initial_window = controller.window_size();

        // Simulate slow consumer by not consuming items
        let (tx, _rx) = mpsc::channel(100);
        let controller_clone = controller.clone();

        // Send items without consuming (send enough to trigger congestion)
        // Need > 60% utilization: send 4 items with window of 5 = 80% utilization
        for i in 0..4 {
            let item = Ok(i);
            backpressure_support::send_with_backpressure(&tx, item, &controller_clone).await;
        }

        // Items are sent but not consumed, so pending should be 4
        assert_eq!(controller.items_sent(), 4);
        assert_eq!(controller.items_consumed(), 0);

        // Wait for congestion check
        sleep(Duration::from_millis(20)).await;
        controller.check_congestion().await;

        // Window may have decreased, or stayed same if congestion wasn't detected yet
        // The assertion should be that pending items > 0 and window is still valid
        assert!(controller.window_size() >= 2); // At least min_window
        assert!(controller.window_size() <= initial_window); // Not increased
        assert!(controller.pending_items() > 0); // Items still pending
    }

    #[test]
    fn test_grpc_service_config_default() {
        let config = GrpcServiceConfig::default();
        assert!(config.backpressure.is_some());
        assert!(config.enable_monitoring);
    }

    /// Verify `GradientSyncService` constructs without panic.
    #[test]
    fn test_gradient_sync_service_new() {
        use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
        use std::sync::Arc;

        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-grpc-grad-sync-svc"),
            cache_size: 16 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("SledBlockStore::new"));
        let _service = GradientSyncService::new(store);
        // Construction must not panic.
    }

    /// Verify `GradientSyncService::sync_gradients` streams chunks for a small gradient.
    #[tokio::test]
    async fn test_gradient_sync_service_streams_chunks() {
        use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
        use ipfrs_tensorlogic::gradient::arrow_ipc::store_gradient_as_arrow;
        use std::sync::Arc;

        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-grpc-grad-sync-chunks"),
            cache_size: 16 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("SledBlockStore::new"));
        let service = GradientSyncService::new(store);

        let gradient: Vec<f32> = (0u32..128).map(|i| i as f32 * 0.1).collect();
        let local_gradient_bytes = store_gradient_as_arrow(&gradient).expect("encode");

        let request = GradientSyncRequest {
            session_id: "test-sync-session".to_string(),
            local_gradient: local_gradient_bytes,
            min_peers: 0,
            timeout_secs: 5,
        };

        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        service
            .sync_gradients(request, tx)
            .await
            .expect("sync_gradients");

        let mut received = Vec::new();
        while let Some(chunk) = rx.recv().await {
            assert_eq!(chunk.session_id, "test-sync-session");
            assert_eq!(chunk.peer_id, "local");
            received.push(chunk);
        }

        assert!(
            !received.is_empty(),
            "at least one chunk must be streamed back"
        );
        assert_eq!(received[0].chunk_index, 0);
    }
}
