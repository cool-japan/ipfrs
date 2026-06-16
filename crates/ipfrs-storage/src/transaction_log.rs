//! ACID-like transaction logging with commit, rollback, and replay support.
//!
//! This module provides [`StorageTransactionLog`], a production-grade transaction
//! log for tracking storage operations with full ACID semantics at the log level:
//!
//! - **Atomicity**: each transaction either commits all operations or none
//! - **Consistency**: state transitions are explicit and guarded
//! - **Isolation**: active transactions are tracked independently
//! - **Durability**: committed entries are retained in a bounded deque for replay
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::transaction_log::{StorageTransactionLog, TxOperation};
//!
//! let mut log = StorageTransactionLog::new(1024);
//! let tx_id = log.begin(0);
//! log.append(tx_id, TxOperation::Put {
//!     key: "foo".to_string(),
//!     value: b"bar".to_vec(),
//!     prev_value: None,
//! }).unwrap();
//! log.commit(tx_id, 1).unwrap();
//! assert_eq!(log.committed_count_total(), 1);
//! ```

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

// ──────────────────────────────────────────────────────────────────────────────
// Public error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`StorageTransactionLog`] operations.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum TxError {
    /// No transaction with the given id exists in active or committed storage.
    #[error("transaction {0} not found")]
    TransactionNotFound(u64),

    /// The transaction exists but is not in the `Active` state.
    #[error("transaction {0} is not active")]
    TransactionNotActive(u64),

    /// The transaction has already been committed, rolled back, or aborted.
    #[error("transaction {0} has already ended")]
    TransactionAlreadyEnded(u64),
}

// ──────────────────────────────────────────────────────────────────────────────
// Core types
// ──────────────────────────────────────────────────────────────────────────────

/// A monotonically increasing transaction identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransactionId(pub u64);

impl TransactionId {
    /// Returns the inner numeric id.
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tx({})", self.0)
    }
}

/// A single operation recorded inside a transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxOperation {
    /// Insert or overwrite a key.  `prev_value` holds the prior content for rollback.
    Put {
        key: String,
        value: Vec<u8>,
        prev_value: Option<Vec<u8>>,
    },
    /// Remove a key.  `prev_value` holds the prior content for rollback.
    Delete { key: String, prev_value: Vec<u8> },
    /// Insert or overwrite multiple keys atomically.
    BatchPut { entries: Vec<(String, Vec<u8>)> },
}

impl TxOperation {
    /// Returns the number of logical key-value pairs affected by this operation.
    pub fn key_count(&self) -> usize {
        match self {
            TxOperation::Put { .. } => 1,
            TxOperation::Delete { .. } => 1,
            TxOperation::BatchPut { entries } => entries.len(),
        }
    }
}

/// Lifecycle state of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransactionStatus {
    /// Transaction is open and accepting operations.
    Active,
    /// Transaction has been successfully committed.
    Committed,
    /// Transaction was rolled back; caller undid the operations.
    RolledBack,
    /// Transaction was aborted without rollback data.
    Aborted,
}

impl std::fmt::Display for TransactionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransactionStatus::Active => write!(f, "Active"),
            TransactionStatus::Committed => write!(f, "Committed"),
            TransactionStatus::RolledBack => write!(f, "RolledBack"),
            TransactionStatus::Aborted => write!(f, "Aborted"),
        }
    }
}

/// A single transaction with its complete operation log and lifecycle metadata.
#[derive(Debug, Clone)]
pub struct Transaction {
    /// Unique identifier assigned at begin time.
    pub id: TransactionId,
    /// Ordered list of operations appended to this transaction.
    pub operations: Vec<TxOperation>,
    /// Current lifecycle state.
    pub status: TransactionStatus,
    /// Logical timestamp (caller-supplied) when the transaction was started.
    pub started_at: u64,
    /// Logical timestamp (caller-supplied) when the transaction ended, if it has.
    pub ended_at: Option<u64>,
}

impl Transaction {
    fn new(id: TransactionId, started_at: u64) -> Self {
        Self {
            id,
            operations: Vec::new(),
            status: TransactionStatus::Active,
            started_at,
            ended_at: None,
        }
    }

    /// Returns `true` if the transaction is still accepting operations.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.status == TransactionStatus::Active
    }

    /// Total number of operations recorded.
    #[inline]
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Aggregate statistics
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics snapshot for a [`StorageTransactionLog`].
#[derive(Debug, Clone, PartialEq)]
pub struct TxStats {
    /// Number of currently active (in-flight) transactions.
    pub active_count: usize,
    /// Total number of transactions that have been committed (cumulative counter).
    pub committed_count: u64,
    /// Total number of transactions that have been rolled back (cumulative counter).
    pub rolled_back_count: u64,
    /// Total operations across all committed transactions currently held in the deque.
    pub total_operations: u64,
    /// Average operations per committed transaction in the deque; `0.0` when deque is empty.
    pub avg_ops_per_tx: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Main struct
// ──────────────────────────────────────────────────────────────────────────────

/// ACID-like transaction log with commit, rollback, and replay support.
///
/// Committed transactions are stored in a bounded [`VecDeque`].  When the deque
/// exceeds `max_committed`, the oldest committed entry is evicted to bound memory
/// usage.  Active transactions live in a separate [`HashMap`] so lookup is O(1).
///
/// All timestamps are *caller-supplied* logical clocks (e.g. milliseconds since
/// epoch), so the struct itself is `Send + Sync` and has no hidden I/O.
pub struct StorageTransactionLog {
    /// Ring buffer of committed transactions, oldest at front.
    transactions: VecDeque<Transaction>,
    /// In-flight transactions indexed by id.
    active: HashMap<TransactionId, Transaction>,
    /// Counter used to hand out the next transaction id.
    next_id: u64,
    /// Maximum number of committed transactions to retain.
    max_committed: usize,
    /// Cumulative committed counter (never decreases).
    committed_count: u64,
    /// Cumulative rolled-back counter (never decreases).
    rolled_back_count: u64,
}

impl StorageTransactionLog {
    // ──────────────────────────────────────────────────────────────────────
    // Construction
    // ──────────────────────────────────────────────────────────────────────

    /// Create a new transaction log.
    ///
    /// `max_committed` controls how many committed [`Transaction`] records are
    /// retained for replay.  A value of `0` means no committed records are
    /// retained (replay will always return an empty list).
    pub fn new(max_committed: usize) -> Self {
        Self {
            transactions: VecDeque::new(),
            active: HashMap::new(),
            next_id: 1,
            max_committed,
            committed_count: 0,
            rolled_back_count: 0,
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Transaction lifecycle
    // ──────────────────────────────────────────────────────────────────────

    /// Begin a new transaction and return its [`TransactionId`].
    ///
    /// `now` is the caller-supplied logical timestamp for `started_at`.
    pub fn begin(&mut self, now: u64) -> TransactionId {
        let id = TransactionId(self.next_id);
        self.next_id += 1;
        let tx = Transaction::new(id, now);
        self.active.insert(id, tx);
        id
    }

    /// Append an operation to an active transaction.
    ///
    /// # Errors
    ///
    /// - [`TxError::TransactionNotFound`] – no transaction with `tx_id` exists
    /// - [`TxError::TransactionNotActive`] – the transaction is no longer active
    pub fn append(&mut self, tx_id: TransactionId, op: TxOperation) -> Result<(), TxError> {
        let tx = self
            .active
            .get_mut(&tx_id)
            .ok_or(TxError::TransactionNotFound(tx_id.0))?;
        if !tx.is_active() {
            return Err(TxError::TransactionNotActive(tx_id.0));
        }
        tx.operations.push(op);
        Ok(())
    }

    /// Commit a transaction, moving it to the committed deque.
    ///
    /// If the deque has reached `max_committed`, the oldest entry is evicted
    /// before the new one is pushed.
    ///
    /// # Errors
    ///
    /// - [`TxError::TransactionNotFound`] – no active transaction with `tx_id`
    /// - [`TxError::TransactionAlreadyEnded`] – the transaction is not active
    pub fn commit(&mut self, tx_id: TransactionId, now: u64) -> Result<(), TxError> {
        let mut tx = self
            .active
            .remove(&tx_id)
            .ok_or(TxError::TransactionNotFound(tx_id.0))?;
        if !tx.is_active() {
            // Put it back so state is consistent, then return error.
            self.active.insert(tx_id, tx);
            return Err(TxError::TransactionAlreadyEnded(tx_id.0));
        }
        tx.status = TransactionStatus::Committed;
        tx.ended_at = Some(now);
        self.committed_count += 1;

        // Evict oldest when we are at capacity.
        if self.max_committed > 0 && self.transactions.len() >= self.max_committed {
            self.transactions.pop_front();
        }
        if self.max_committed > 0 {
            self.transactions.push_back(tx);
        }
        // If max_committed == 0 the tx is simply discarded (not retained).
        Ok(())
    }

    /// Roll back a transaction, returning its operations in **reverse** order.
    ///
    /// The caller is expected to use the returned operations to undo changes
    /// (e.g. restore `prev_value` for `Put`/`Delete` entries).
    ///
    /// # Errors
    ///
    /// - [`TxError::TransactionNotFound`] – no active transaction with `tx_id`
    /// - [`TxError::TransactionAlreadyEnded`] – the transaction is not active
    pub fn rollback(
        &mut self,
        tx_id: TransactionId,
        now: u64,
    ) -> Result<Vec<TxOperation>, TxError> {
        let mut tx = self
            .active
            .remove(&tx_id)
            .ok_or(TxError::TransactionNotFound(tx_id.0))?;
        if !tx.is_active() {
            self.active.insert(tx_id, tx);
            return Err(TxError::TransactionAlreadyEnded(tx_id.0));
        }
        tx.status = TransactionStatus::RolledBack;
        tx.ended_at = Some(now);
        self.rolled_back_count += 1;

        // Return operations in reverse order for undo.
        let mut ops = tx.operations.clone();
        ops.reverse();
        // We intentionally do not retain rolled-back transactions in the replay
        // deque because they were never durably applied.
        Ok(ops)
    }

    /// Abort a transaction without providing rollback data to the caller.
    ///
    /// Unlike [`rollback`](Self::rollback), this method discards the operations
    /// and does **not** return them.  Use this when the transaction state is
    /// already inconsistent or when rollback undo is handled elsewhere.
    ///
    /// # Errors
    ///
    /// - [`TxError::TransactionNotFound`] – no active transaction with `tx_id`
    /// - [`TxError::TransactionAlreadyEnded`] – the transaction is not active
    pub fn abort(&mut self, tx_id: TransactionId, now: u64) -> Result<(), TxError> {
        let mut tx = self
            .active
            .remove(&tx_id)
            .ok_or(TxError::TransactionNotFound(tx_id.0))?;
        if !tx.is_active() {
            self.active.insert(tx_id, tx);
            return Err(TxError::TransactionAlreadyEnded(tx_id.0));
        }
        tx.status = TransactionStatus::Aborted;
        tx.ended_at = Some(now);
        // Aborted transactions are not counted in rolled_back_count and are not
        // retained.
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    // Query / replay
    // ──────────────────────────────────────────────────────────────────────

    /// Returns all committed transactions whose id is **strictly greater** than
    /// `since_id`, in ascending id order.
    ///
    /// This is the primary replay API: supply `TransactionId(0)` to replay from
    /// the beginning of the retained window.
    pub fn replay_committed(&self, since_id: TransactionId) -> Vec<&Transaction> {
        self.transactions
            .iter()
            .filter(|tx| tx.id > since_id)
            .collect()
    }

    /// Look up a transaction by id.
    ///
    /// Searches the active map first, then the committed deque.  Returns `None`
    /// if the transaction has been evicted or was never created.
    pub fn get_transaction(&self, tx_id: TransactionId) -> Option<&Transaction> {
        if let Some(tx) = self.active.get(&tx_id) {
            return Some(tx);
        }
        self.transactions.iter().find(|tx| tx.id == tx_id)
    }

    /// Returns references to all currently active (in-flight) transactions,
    /// sorted by id for deterministic ordering.
    pub fn active_transactions(&self) -> Vec<&Transaction> {
        let mut txs: Vec<&Transaction> = self.active.values().collect();
        txs.sort_by_key(|tx| tx.id);
        txs
    }

    /// Number of currently active transactions.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Cumulative count of all committed transactions since this log was created.
    #[inline]
    pub fn committed_count_total(&self) -> u64 {
        self.committed_count
    }

    /// Cumulative count of all rolled-back transactions since this log was created.
    #[inline]
    pub fn rolled_back_count_total(&self) -> u64 {
        self.rolled_back_count
    }

    /// Returns a statistics snapshot.
    ///
    /// `total_operations` counts the sum of operations across all committed
    /// transactions currently held in the deque (not across all historical
    /// transactions).
    pub fn stats(&self) -> TxStats {
        let total_operations: u64 = self
            .transactions
            .iter()
            .map(|tx| tx.operations.len() as u64)
            .sum();
        let deque_len = self.transactions.len();
        let avg_ops_per_tx = if deque_len == 0 {
            0.0
        } else {
            total_operations as f64 / deque_len as f64
        };
        TxStats {
            active_count: self.active.len(),
            committed_count: self.committed_count,
            rolled_back_count: self.rolled_back_count,
            total_operations,
            avg_ops_per_tx,
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Utility
    // ──────────────────────────────────────────────────────────────────────

    /// Returns the number of committed transactions currently in the deque.
    #[inline]
    pub fn committed_retained(&self) -> usize {
        self.transactions.len()
    }

    /// Returns the configured maximum number of retained committed transactions.
    #[inline]
    pub fn max_committed(&self) -> usize {
        self.max_committed
    }

    /// Returns the id that will be assigned to the **next** transaction.
    #[inline]
    pub fn next_transaction_id(&self) -> TransactionId {
        TransactionId(self.next_id)
    }

    /// Drain all committed transactions from the deque and return them.
    ///
    /// This can be used to persist committed transactions to durable storage.
    pub fn drain_committed(&mut self) -> Vec<Transaction> {
        self.transactions.drain(..).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::transaction_log::{
        StorageTransactionLog, TransactionId, TransactionStatus, TxError, TxOperation, TxStats,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn put_op(key: &str, value: &[u8]) -> TxOperation {
        TxOperation::Put {
            key: key.to_string(),
            value: value.to_vec(),
            prev_value: None,
        }
    }

    fn put_op_with_prev(key: &str, value: &[u8], prev: &[u8]) -> TxOperation {
        TxOperation::Put {
            key: key.to_string(),
            value: value.to_vec(),
            prev_value: Some(prev.to_vec()),
        }
    }

    fn delete_op(key: &str, prev: &[u8]) -> TxOperation {
        TxOperation::Delete {
            key: key.to_string(),
            prev_value: prev.to_vec(),
        }
    }

    fn batch_put_op(entries: &[(&str, &[u8])]) -> TxOperation {
        TxOperation::BatchPut {
            entries: entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_vec()))
                .collect(),
        }
    }

    // ── 1. Construction ───────────────────────────────────────────────────────

    #[test]
    fn test_new_log_is_empty() {
        let log = StorageTransactionLog::new(100);
        assert_eq!(log.active_count(), 0);
        assert_eq!(log.committed_count_total(), 0);
        assert_eq!(log.rolled_back_count_total(), 0);
        assert_eq!(log.committed_retained(), 0);
    }

    #[test]
    fn test_max_committed_preserved() {
        let log = StorageTransactionLog::new(42);
        assert_eq!(log.max_committed(), 42);
    }

    #[test]
    fn test_next_id_starts_at_one() {
        let log = StorageTransactionLog::new(10);
        assert_eq!(log.next_transaction_id(), TransactionId(1));
    }

    // ── 2. begin ─────────────────────────────────────────────────────────────

    #[test]
    fn test_begin_returns_incrementing_ids() {
        let mut log = StorageTransactionLog::new(10);
        let id1 = log.begin(0);
        let id2 = log.begin(1);
        let id3 = log.begin(2);
        assert_eq!(id1, TransactionId(1));
        assert_eq!(id2, TransactionId(2));
        assert_eq!(id3, TransactionId(3));
    }

    #[test]
    fn test_begin_increments_active_count() {
        let mut log = StorageTransactionLog::new(10);
        log.begin(0);
        assert_eq!(log.active_count(), 1);
        log.begin(0);
        assert_eq!(log.active_count(), 2);
    }

    #[test]
    fn test_begin_transaction_is_active_status() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(100);
        let tx = log.get_transaction(id).expect("tx must exist");
        assert_eq!(tx.status, TransactionStatus::Active);
        assert_eq!(tx.started_at, 100);
        assert!(tx.ended_at.is_none());
    }

    // ── 3. append ────────────────────────────────────────────────────────────

    #[test]
    fn test_append_put_operation() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        let result = log.append(id, put_op("k", b"v"));
        assert!(result.is_ok());
        let tx = log.get_transaction(id).expect("tx must exist");
        assert_eq!(tx.operations.len(), 1);
    }

    #[test]
    fn test_append_multiple_operations() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.append(id, put_op("a", b"1")).unwrap();
        log.append(id, put_op("b", b"2")).unwrap();
        log.append(id, delete_op("c", b"old")).unwrap();
        let tx = log.get_transaction(id).expect("tx must exist");
        assert_eq!(tx.operations.len(), 3);
    }

    #[test]
    fn test_append_unknown_tx_returns_not_found() {
        let mut log = StorageTransactionLog::new(10);
        let err = log
            .append(TransactionId(99), put_op("k", b"v"))
            .unwrap_err();
        assert_eq!(err, TxError::TransactionNotFound(99));
    }

    #[test]
    fn test_append_batch_put_operation() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        let op = batch_put_op(&[("x", b"1"), ("y", b"2"), ("z", b"3")]);
        log.append(id, op).unwrap();
        let tx = log.get_transaction(id).unwrap();
        assert_eq!(tx.operations.len(), 1);
        if let TxOperation::BatchPut { entries } = &tx.operations[0] {
            assert_eq!(entries.len(), 3);
        } else {
            panic!("expected BatchPut");
        }
    }

    // ── 4. commit ────────────────────────────────────────────────────────────

    #[test]
    fn test_commit_moves_tx_to_committed() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.append(id, put_op("k", b"v")).unwrap();
        log.commit(id, 1).unwrap();
        assert_eq!(log.active_count(), 0);
        assert_eq!(log.committed_count_total(), 1);
        assert_eq!(log.committed_retained(), 1);
    }

    #[test]
    fn test_commit_sets_ended_at() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.commit(id, 42).unwrap();
        let tx = log.get_transaction(id).expect("committed tx visible");
        assert_eq!(tx.ended_at, Some(42));
        assert_eq!(tx.status, TransactionStatus::Committed);
    }

    #[test]
    fn test_commit_unknown_tx_returns_not_found() {
        let mut log = StorageTransactionLog::new(10);
        let err = log.commit(TransactionId(77), 0).unwrap_err();
        assert_eq!(err, TxError::TransactionNotFound(77));
    }

    #[test]
    fn test_commit_evicts_oldest_when_at_capacity() {
        let mut log = StorageTransactionLog::new(3);
        let ids: Vec<_> = (0..4).map(|t| log.begin(t)).collect();
        for (i, &id) in ids.iter().enumerate() {
            log.commit(id, i as u64 + 10).unwrap();
        }
        assert_eq!(log.committed_retained(), 3);
        // Oldest (id=1) should have been evicted.
        assert!(log.get_transaction(TransactionId(1)).is_none());
        // Others remain.
        assert!(log.get_transaction(TransactionId(2)).is_some());
        assert!(log.get_transaction(TransactionId(3)).is_some());
        assert!(log.get_transaction(TransactionId(4)).is_some());
    }

    #[test]
    fn test_commit_with_max_zero_retains_nothing() {
        let mut log = StorageTransactionLog::new(0);
        let id = log.begin(0);
        log.commit(id, 1).unwrap();
        assert_eq!(log.committed_retained(), 0);
        assert_eq!(log.committed_count_total(), 1);
    }

    // ── 5. rollback ──────────────────────────────────────────────────────────

    #[test]
    fn test_rollback_returns_ops_in_reverse() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.append(id, put_op("a", b"1")).unwrap();
        log.append(id, put_op("b", b"2")).unwrap();
        log.append(id, delete_op("c", b"old")).unwrap();
        let ops = log.rollback(id, 5).unwrap();
        assert_eq!(ops.len(), 3);
        // Last appended op should be first in the returned list.
        assert_eq!(ops[0], delete_op("c", b"old"));
        assert_eq!(ops[1], put_op("b", b"2"));
        assert_eq!(ops[2], put_op("a", b"1"));
    }

    #[test]
    fn test_rollback_increments_rolled_back_count() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.rollback(id, 1).unwrap();
        assert_eq!(log.rolled_back_count_total(), 1);
    }

    #[test]
    fn test_rollback_removes_from_active() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.rollback(id, 1).unwrap();
        assert_eq!(log.active_count(), 0);
    }

    #[test]
    fn test_rollback_does_not_add_to_committed_deque() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.rollback(id, 1).unwrap();
        assert_eq!(log.committed_retained(), 0);
    }

    #[test]
    fn test_rollback_unknown_tx_returns_not_found() {
        let mut log = StorageTransactionLog::new(10);
        let err = log.rollback(TransactionId(5), 0).unwrap_err();
        assert_eq!(err, TxError::TransactionNotFound(5));
    }

    #[test]
    fn test_rollback_empty_tx_returns_empty_ops() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        let ops = log.rollback(id, 1).unwrap();
        assert!(ops.is_empty());
    }

    // ── 6. abort ─────────────────────────────────────────────────────────────

    #[test]
    fn test_abort_removes_from_active() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.abort(id, 1).unwrap();
        assert_eq!(log.active_count(), 0);
    }

    #[test]
    fn test_abort_does_not_increment_committed_count() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.abort(id, 1).unwrap();
        assert_eq!(log.committed_count_total(), 0);
    }

    #[test]
    fn test_abort_does_not_increment_rolled_back_count() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.abort(id, 1).unwrap();
        assert_eq!(log.rolled_back_count_total(), 0);
    }

    #[test]
    fn test_abort_unknown_tx_returns_not_found() {
        let mut log = StorageTransactionLog::new(10);
        let err = log.abort(TransactionId(9), 0).unwrap_err();
        assert_eq!(err, TxError::TransactionNotFound(9));
    }

    #[test]
    fn test_abort_does_not_retain_in_deque() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.abort(id, 1).unwrap();
        assert_eq!(log.committed_retained(), 0);
    }

    // ── 7. replay_committed ──────────────────────────────────────────────────

    #[test]
    fn test_replay_committed_since_zero_returns_all() {
        let mut log = StorageTransactionLog::new(10);
        let id1 = log.begin(0);
        log.commit(id1, 1).unwrap();
        let id2 = log.begin(2);
        log.commit(id2, 3).unwrap();
        let replayed = log.replay_committed(TransactionId(0));
        assert_eq!(replayed.len(), 2);
    }

    #[test]
    fn test_replay_committed_filters_by_id() {
        let mut log = StorageTransactionLog::new(10);
        let id1 = log.begin(0);
        log.commit(id1, 1).unwrap();
        let id2 = log.begin(2);
        log.commit(id2, 3).unwrap();
        let id3 = log.begin(4);
        log.commit(id3, 5).unwrap();
        // Replay only after id1 (i.e., id ≥ 2)
        let replayed = log.replay_committed(id1);
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].id, id2);
        assert_eq!(replayed[1].id, id3);
    }

    #[test]
    fn test_replay_committed_returns_empty_for_high_since_id() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.commit(id, 1).unwrap();
        let replayed = log.replay_committed(TransactionId(999));
        assert!(replayed.is_empty());
    }

    #[test]
    fn test_replay_committed_order_is_ascending() {
        let mut log = StorageTransactionLog::new(10);
        for t in 0..5u64 {
            let id = log.begin(t);
            log.commit(id, t + 1).unwrap();
        }
        let replayed = log.replay_committed(TransactionId(0));
        let ids: Vec<u64> = replayed.iter().map(|tx| tx.id.0).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    // ── 8. get_transaction ───────────────────────────────────────────────────

    #[test]
    fn test_get_transaction_active() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(7);
        let tx = log.get_transaction(id).expect("should find active tx");
        assert_eq!(tx.id, id);
        assert_eq!(tx.status, TransactionStatus::Active);
    }

    #[test]
    fn test_get_transaction_committed() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.commit(id, 1).unwrap();
        let tx = log.get_transaction(id).expect("should find committed tx");
        assert_eq!(tx.status, TransactionStatus::Committed);
    }

    #[test]
    fn test_get_transaction_returns_none_for_unknown() {
        let log = StorageTransactionLog::new(10);
        assert!(log.get_transaction(TransactionId(42)).is_none());
    }

    // ── 9. active_transactions ───────────────────────────────────────────────

    #[test]
    fn test_active_transactions_sorted_by_id() {
        let mut log = StorageTransactionLog::new(10);
        let _id3 = log.begin(2);
        let _id1 = log.begin(0);
        let _id2 = log.begin(1);
        let active = log.active_transactions();
        let ids: Vec<u64> = active.iter().map(|tx| tx.id.0).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn test_active_transactions_empty_after_all_committed() {
        let mut log = StorageTransactionLog::new(10);
        let ids: Vec<_> = (0..3).map(|t| log.begin(t)).collect();
        for id in ids {
            log.commit(id, 99).unwrap();
        }
        assert!(log.active_transactions().is_empty());
    }

    // ── 10. stats ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let log = StorageTransactionLog::new(10);
        let s = log.stats();
        assert_eq!(
            s,
            TxStats {
                active_count: 0,
                committed_count: 0,
                rolled_back_count: 0,
                total_operations: 0,
                avg_ops_per_tx: 0.0,
            }
        );
    }

    #[test]
    fn test_stats_reflects_active_and_committed() {
        let mut log = StorageTransactionLog::new(10);
        let id1 = log.begin(0);
        log.append(id1, put_op("a", b"1")).unwrap();
        log.append(id1, put_op("b", b"2")).unwrap();
        log.commit(id1, 1).unwrap();

        let _id2 = log.begin(2);

        let s = log.stats();
        assert_eq!(s.active_count, 1);
        assert_eq!(s.committed_count, 1);
        assert_eq!(s.rolled_back_count, 0);
        assert_eq!(s.total_operations, 2);
        assert!((s.avg_ops_per_tx - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_avg_ops_zero_when_no_committed_in_deque() {
        let mut log = StorageTransactionLog::new(0); // no retention
        let id = log.begin(0);
        log.append(id, put_op("k", b"v")).unwrap();
        log.commit(id, 1).unwrap();
        let s = log.stats();
        assert_eq!(s.avg_ops_per_tx, 0.0);
        assert_eq!(s.total_operations, 0);
    }

    // ── 11. TxOperation helpers ──────────────────────────────────────────────

    #[test]
    fn test_tx_operation_key_count_put() {
        assert_eq!(put_op("k", b"v").key_count(), 1);
    }

    #[test]
    fn test_tx_operation_key_count_delete() {
        assert_eq!(delete_op("k", b"v").key_count(), 1);
    }

    #[test]
    fn test_tx_operation_key_count_batch_put() {
        let op = batch_put_op(&[("a", b"1"), ("b", b"2"), ("c", b"3"), ("d", b"4")]);
        assert_eq!(op.key_count(), 4);
    }

    // ── 12. drain_committed ───────────────────────────────────────────────────

    #[test]
    fn test_drain_committed_returns_and_clears() {
        let mut log = StorageTransactionLog::new(10);
        let id1 = log.begin(0);
        log.commit(id1, 1).unwrap();
        let id2 = log.begin(2);
        log.commit(id2, 3).unwrap();
        let drained = log.drain_committed();
        assert_eq!(drained.len(), 2);
        assert_eq!(log.committed_retained(), 0);
    }

    // ── 13. concurrent-ish interleaving ──────────────────────────────────────

    #[test]
    fn test_interleaved_transactions() {
        let mut log = StorageTransactionLog::new(10);
        let id_a = log.begin(0);
        let id_b = log.begin(0);
        log.append(id_a, put_op("a", b"va")).unwrap();
        log.append(id_b, put_op("b", b"vb")).unwrap();
        log.append(id_a, put_op("a2", b"va2")).unwrap();
        log.commit(id_b, 1).unwrap();
        log.rollback(id_a, 2).unwrap();
        assert_eq!(log.committed_count_total(), 1);
        assert_eq!(log.rolled_back_count_total(), 1);
        assert_eq!(log.active_count(), 0);
    }

    #[test]
    fn test_rollback_with_prev_values() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        log.append(id, put_op_with_prev("k", b"new", b"old"))
            .unwrap();
        let ops = log.rollback(id, 1).unwrap();
        assert_eq!(ops.len(), 1);
        if let TxOperation::Put { prev_value, .. } = &ops[0] {
            assert_eq!(prev_value.as_deref(), Some(b"old".as_slice()));
        } else {
            panic!("expected Put op");
        }
    }

    // ── 14. edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_commit_does_not_affect_other_active_txs() {
        let mut log = StorageTransactionLog::new(10);
        let id_a = log.begin(0);
        let id_b = log.begin(0);
        log.commit(id_a, 1).unwrap();
        assert_eq!(log.active_count(), 1);
        assert!(log.get_transaction(id_b).is_some());
    }

    #[test]
    fn test_large_batch_eviction_boundary() {
        let mut log = StorageTransactionLog::new(5);
        let ids: Vec<_> = (0..10u64).map(|t| log.begin(t)).collect();
        for (i, &id) in ids.iter().enumerate() {
            log.commit(id, i as u64 + 100).unwrap();
        }
        assert_eq!(log.committed_retained(), 5);
        assert_eq!(log.committed_count_total(), 10);
        // Only the last 5 should be retained (ids 6–10).
        let retained = log.replay_committed(TransactionId(0));
        let min_id = retained.iter().map(|tx| tx.id.0).min().unwrap_or(0);
        assert_eq!(min_id, 6);
    }

    #[test]
    fn test_transaction_id_ordering() {
        assert!(TransactionId(1) < TransactionId(2));
        assert!(TransactionId(100) > TransactionId(50));
        assert_eq!(TransactionId(5), TransactionId(5));
    }

    #[test]
    fn test_transaction_id_display() {
        let id = TransactionId(7);
        assert_eq!(format!("{id}"), "Tx(7)");
    }

    #[test]
    fn test_tx_error_display() {
        assert_eq!(
            format!("{}", TxError::TransactionNotFound(3)),
            "transaction 3 not found"
        );
        assert_eq!(
            format!("{}", TxError::TransactionNotActive(5)),
            "transaction 5 is not active"
        );
        assert_eq!(
            format!("{}", TxError::TransactionAlreadyEnded(9)),
            "transaction 9 has already ended"
        );
    }

    #[test]
    fn test_many_operations_in_single_tx() {
        let mut log = StorageTransactionLog::new(10);
        let id = log.begin(0);
        for i in 0u64..500 {
            log.append(id, put_op(&format!("key-{i}"), b"value"))
                .unwrap();
        }
        let tx = log.get_transaction(id).unwrap();
        assert_eq!(tx.operation_count(), 500);
        log.commit(id, 1).unwrap();
        let s = log.stats();
        assert_eq!(s.total_operations, 500);
    }
}
