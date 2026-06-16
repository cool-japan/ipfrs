//! DAG (Directed Acyclic Graph) operations for Node

use bytes::Bytes;
use ipfrs_core::{Block, Cid, Error, Result};
use ipfrs_storage::BlockStoreTrait;
use std::path::Path;

use super::{DagExportStats, DagImportStats, Node};

impl Node {
    /// Store an IPLD DAG node
    ///
    /// Serializes the IPLD data structure, stores it as a block, and returns the CID.
    /// This is useful for storing structured data with links to other blocks.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    /// use ipfrs_core::Ipld;
    /// use std::collections::BTreeMap;
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert("name".to_string(), Ipld::String("Alice".to_string()));
    /// map.insert("age".to_string(), Ipld::Integer(30));
    ///
    /// let cid = node.dag_put(Ipld::Map(map)).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn dag_put(&self, data: ipfrs_core::Ipld) -> Result<Cid> {
        let storage = self.storage()?;

        // Serialize IPLD to DAG-CBOR format
        let bytes = data.to_dag_cbor()?;

        // Create block and store
        let block = Block::new(Bytes::from(bytes))?;
        let cid = *block.cid();

        storage.put(&block).await?;

        Ok(cid)
    }

    /// Retrieve an IPLD DAG node
    ///
    /// Fetches a block by CID and deserializes it as IPLD.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(data) = node.dag_get(&cid).await? {
    ///     println!("Retrieved DAG node: {:?}", data);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn dag_get(&self, cid: &Cid) -> Result<Option<ipfrs_core::Ipld>> {
        let storage = self.storage()?;

        match storage.get(cid).await? {
            Some(block) => {
                let ipld = ipfrs_core::Ipld::from_dag_cbor(block.data())?;
                Ok(Some(ipld))
            }
            None => Ok(None),
        }
    }

    /// Resolve an IPLD path
    ///
    /// Navigates through IPLD structures following a path like "/key1/key2/0".
    /// Returns the CID if the path leads to a link.
    ///
    /// # Path Format
    /// - Map keys: "/key"
    /// - List indices: "/0", "/1", etc.
    /// - Nested: "/map_key/0/nested_key"
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let root_cid = ipfrs_core::Cid::default();
    /// // Resolve path: /users/0/profile
    /// if let Some(cid) = node.dag_resolve(&root_cid, "/users/0/profile").await? {
    ///     println!("Resolved to CID: {}", cid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn dag_resolve(&self, root: &Cid, path: &str) -> Result<Option<Cid>> {
        let mut current_cid = *root;
        let parts: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        if parts.is_empty() {
            return Ok(Some(current_cid));
        }

        for part in parts {
            // Get the current node
            let ipld = match self.dag_get(&current_cid).await? {
                Some(ipld) => ipld,
                None => return Ok(None),
            };

            // Navigate to the next node
            match ipld {
                ipfrs_core::Ipld::Map(map) => {
                    match map.get(part) {
                        Some(ipfrs_core::Ipld::Link(link)) => {
                            current_cid = link.0;
                        }
                        Some(_value) => {
                            // For non-link values, we can't resolve further
                            return Err(Error::InvalidData(format!(
                                "Path leads to non-link value at '{}'",
                                part
                            )));
                        }
                        None => {
                            return Ok(None);
                        }
                    }
                }
                ipfrs_core::Ipld::List(list) => {
                    let index: usize = part.parse().map_err(|_| {
                        Error::InvalidData(format!("Invalid list index: '{}'", part))
                    })?;

                    match list.get(index) {
                        Some(ipfrs_core::Ipld::Link(link)) => {
                            current_cid = link.0;
                        }
                        Some(_) => {
                            return Err(Error::InvalidData(format!(
                                "Path leads to non-link value at index {}",
                                index
                            )));
                        }
                        None => {
                            return Ok(None);
                        }
                    }
                }
                _ => {
                    return Err(Error::InvalidData(format!(
                        "Cannot navigate through non-map/non-list value at '{}'",
                        part
                    )));
                }
            }
        }

        Ok(Some(current_cid))
    }

    /// Traverse a DAG and collect all reachable CIDs
    ///
    /// Performs a breadth-first traversal starting from the root CID,
    /// following all links and collecting all reachable CIDs.
    ///
    /// # Parameters
    /// - `root`: The starting CID
    /// - `max_depth`: Maximum depth to traverse (None for unlimited)
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let root_cid = ipfrs_core::Cid::default();
    /// // Traverse up to depth 10
    /// let cids = node.dag_traverse(&root_cid, Some(10)).await?;
    /// println!("Found {} reachable blocks", cids.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn dag_traverse(&self, root: &Cid, max_depth: Option<usize>) -> Result<Vec<Cid>> {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((*root, 0usize));
        visited.insert(*root);

        while let Some((cid, depth)) = queue.pop_front() {
            // Check depth limit
            if let Some(max) = max_depth {
                if depth >= max {
                    continue;
                }
            }

            result.push(cid);

            // Get the node
            if let Some(ipld) = self.dag_get(&cid).await? {
                // Extract all links
                for link_cid in ipld.links() {
                    if visited.insert(link_cid) {
                        queue.push_back((link_cid, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Export a DAG to CAR (Content Addressable aRchive) format
    pub async fn dag_export(&self, root: &Cid, path: impl AsRef<Path>) -> Result<DagExportStats> {
        use ipfrs_storage::export_to_car;

        let storage = self.storage()?;
        let roots = vec![*root];

        let stats = export_to_car(storage.as_ref(), path.as_ref(), roots).await?;

        Ok(DagExportStats {
            blocks_exported: stats.blocks_written,
            bytes_exported: stats.bytes_written,
        })
    }

    /// Import blocks from a CAR file
    pub async fn dag_import(&self, path: impl AsRef<Path>) -> Result<DagImportStats> {
        use ipfrs_storage::import_from_car;

        let storage = self.storage()?;
        let stats = import_from_car(storage.as_ref(), path.as_ref()).await?;

        Ok(DagImportStats {
            blocks_imported: stats.blocks_read,
            bytes_imported: stats.bytes_read,
        })
    }
}
