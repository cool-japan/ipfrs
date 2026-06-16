//! Pin management operations for Node

use ipfrs_core::{Cid, Result};
use ipfrs_storage::BlockStoreTrait;
use std::path::Path;

use crate::pin::{PinInfo, PinType};

use super::Node;

impl Node {
    /// Pin a block (prevent it from being garbage collected)
    pub async fn pin_add(&self, cid: &Cid, recursive: bool, name: Option<String>) -> Result<()> {
        // Verify the block exists
        if !self.storage()?.has(cid).await? {
            return Err(ipfrs_core::Error::NotFound(cid.to_string()));
        }

        let pin_type = if recursive {
            PinType::Recursive
        } else {
            PinType::Direct
        };

        self.pin_manager.pin(*cid, pin_type, name)?;

        // If recursive, traverse and mark all referenced blocks
        if recursive {
            self.mark_recursive_pins(cid).await?;
        }

        Ok(())
    }

    /// Mark all blocks referenced by a CID as indirectly pinned
    pub(super) async fn mark_recursive_pins(&self, root: &Cid) -> Result<()> {
        let storage = self.storage()?;
        let mut to_visit = vec![*root];
        let mut visited = std::collections::HashSet::new();

        while let Some(cid) = to_visit.pop() {
            if !visited.insert(cid) {
                continue;
            }

            // Get the block and try to parse as IPLD
            if let Some(block) = storage.get(&cid).await? {
                if let Ok(ipld) = ipfrs_core::Ipld::from_dag_cbor(block.data()) {
                    // Add all links as indirect pins
                    for link_cid in ipld.links() {
                        self.pin_manager.add_indirect(*root, link_cid);
                        to_visit.push(link_cid);
                    }
                }
            }
        }

        Ok(())
    }

    /// Unpin a block
    pub async fn pin_rm(&self, cid: &Cid, recursive: bool) -> Result<()> {
        self.pin_manager.unpin(cid, recursive)
    }

    /// List all pinned blocks
    pub fn pin_ls(&self) -> Result<Vec<PinInfo>> {
        Ok(self.pin_manager.list())
    }

    /// Verify all pins are available
    pub async fn pin_verify(&self) -> Result<Vec<(Cid, bool)>> {
        let storage = self.storage()?;
        let pins = self.pin_manager.list();
        let mut results = Vec::new();

        for pin in pins {
            let exists = storage.has(&pin.cid).await?;
            results.push((pin.cid, exists));
        }

        Ok(results)
    }

    /// Save pin index to disk
    pub async fn pin_save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.pin_manager.save(path).await
    }

    /// Load pin index from disk
    pub async fn pin_load(&self, path: impl AsRef<Path>) -> Result<()> {
        self.pin_manager.load(path).await
    }
}
