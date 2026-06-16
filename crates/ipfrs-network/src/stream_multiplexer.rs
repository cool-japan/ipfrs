//! Stream Multiplexer — multiplexes multiple logical streams over a single connection
//! with flow control, priority scheduling, and proper frame lifecycle management.
//!
//! ## Design
//!
//! Each logical [`StreamId`] carries its own sequence counter, send/receive windows,
//! and priority. The shared send queue is a max-[`BinaryHeap`] keyed on
//! `(priority_weight, Reverse(enqueue_sequence))` so that within the same priority
//! tier frames are emitted in FIFO order.
//!
//! ## Frame flags (FrameFlags bit positions)
//!
//! | Bit  | Name | Value | Meaning                        |
//! |------|------|-------|-------------------------------|
//! | 0    | SYN  | 0x01  | Open (create) a new stream     |
//! | 1    | FIN  | 0x02  | Close stream after this frame  |
//! | 2    | RST  | 0x04  | Reset stream immediately       |
//! | 3    | ACK  | 0x08  | Acknowledgement                |
//! | 4    | DATA | 0x10  | Frame carries payload data     |
//!
//! ## Legacy flag constants (raw u8)
//!
//! For backwards compatibility the original raw constants are preserved:
//! `FLAG_FIN=0x01`, `FLAG_RST=0x02`, `FLAG_SYN=0x04`.
//!
//! ## Flow control
//!
//! `send_window` tracks how many bytes may still be enqueued for sending on a
//! given stream.  Each call to [`StreamMultiplexer::send`] checks that
//! `data.len() <= send_window` and deducts the amount from the window.
//! [`StreamMultiplexer::update_window`] adds credits back (simulating receipt of
//! a window-update ACK from the remote side).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Legacy raw flag constants (backward-compatible; raw u8)
// ─────────────────────────────────────────────────────────────────────────────

/// Bit mask for the FIN flag in [`StreamFrame::flags`] (legacy raw u8 form).
pub const FLAG_FIN: u8 = 0b0000_0001;
/// Bit mask for the RST flag in [`StreamFrame::flags`] (legacy raw u8 form).
pub const FLAG_RST: u8 = 0b0000_0010;
/// Bit mask for the SYN flag in [`StreamFrame::flags`] (legacy raw u8 form).
pub const FLAG_SYN: u8 = 0b0000_0100;

// ─────────────────────────────────────────────────────────────────────────────
// FrameFlags — typed bitfield
// ─────────────────────────────────────────────────────────────────────────────

/// Bitfield wrapper for stream-frame control flags.
///
/// # Flag layout
///
/// | Bit | Constant              | Meaning                        |
/// |-----|-----------------------|-------------------------------|
/// | 0   | [`FrameFlags::SYN`]   | Open (create) a new stream     |
/// | 1   | [`FrameFlags::FIN`]   | Close stream gracefully        |
/// | 2   | [`FrameFlags::RST`]   | Reset stream immediately       |
/// | 3   | [`FrameFlags::ACK`]   | Acknowledgement frame          |
/// | 4   | [`FrameFlags::DATA`]  | Frame carries payload data     |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FrameFlags(pub u8);

impl FrameFlags {
    /// SYN — open a new stream (bit 0).
    pub const SYN: u8 = 0x01;
    /// FIN — close stream gracefully (bit 1).
    pub const FIN: u8 = 0x02;
    /// RST — reset stream immediately (bit 2).
    pub const RST: u8 = 0x04;
    /// ACK — acknowledgement (bit 3).
    pub const ACK: u8 = 0x08;
    /// DATA — frame carries payload (bit 4).
    pub const DATA: u8 = 0x10;

    /// Create `FrameFlags` from a raw byte.
    #[inline]
    pub fn new(raw: u8) -> Self {
        Self(raw)
    }

    /// Returns `true` if the SYN bit is set.
    #[inline]
    pub fn is_syn(self) -> bool {
        self.0 & Self::SYN != 0
    }

    /// Returns `true` if the FIN bit is set.
    #[inline]
    pub fn is_fin(self) -> bool {
        self.0 & Self::FIN != 0
    }

    /// Returns `true` if the RST bit is set.
    #[inline]
    pub fn is_rst(self) -> bool {
        self.0 & Self::RST != 0
    }

    /// Returns `true` if the ACK bit is set.
    #[inline]
    pub fn is_ack(self) -> bool {
        self.0 & Self::ACK != 0
    }

    /// Returns `true` if the DATA bit is set.
    #[inline]
    pub fn is_data(self) -> bool {
        self.0 & Self::DATA != 0
    }

    /// Set a flag bit and return the updated value.
    #[inline]
    pub fn with(self, flag: u8) -> Self {
        Self(self.0 | flag)
    }
}

impl fmt::Display for FrameFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<&str> = Vec::new();
        if self.is_syn() {
            parts.push("SYN");
        }
        if self.is_fin() {
            parts.push("FIN");
        }
        if self.is_rst() {
            parts.push("RST");
        }
        if self.is_ack() {
            parts.push("ACK");
        }
        if self.is_data() {
            parts.push("DATA");
        }
        if parts.is_empty() {
            write!(f, "NONE")
        } else {
            write!(f, "{}", parts.join("|"))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamId
// ─────────────────────────────────────────────────────────────────────────────

/// Newtype identifier for a logical stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamId(pub u32);

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream:{}", self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamPriority
// ─────────────────────────────────────────────────────────────────────────────

/// Priority level for a logical stream.
///
/// Higher priority streams are dequeued first by the multiplexer.  Within the
/// same priority, frames are emitted in FIFO order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    /// Best-effort delivery — weight 0.
    Background = 0,
    /// Low priority — weight 1.
    Low = 1,
    /// Normal priority — weight 2.
    Normal = 2,
    /// High priority — weight 4.
    High = 4,
    /// Critical (real-time) — weight 8.
    Critical = 8,
}

impl StreamPriority {
    /// Numeric weight used for priority scheduling.
    ///
    /// `Critical` → 8, `High` → 4, `Normal` → 2, `Low` → 1, `Background` → 0.
    #[inline]
    pub fn weight(self) -> u32 {
        match self {
            Self::Critical => 8,
            Self::High => 4,
            Self::Normal => 2,
            Self::Low => 1,
            Self::Background => 0,
        }
    }
}

/// Convert a raw priority byte (0-255) to the nearest [`StreamPriority`] tier.
pub fn priority_from_u8(p: u8) -> StreamPriority {
    match p {
        0 => StreamPriority::Background,
        1..=63 => StreamPriority::Low,
        64..=127 => StreamPriority::Normal,
        128..=191 => StreamPriority::High,
        192..=255 => StreamPriority::Critical,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamState
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of a logical stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// SYN has been sent but not yet acknowledged; stream is being established.
    Opening,
    /// The stream is fully open and can send/receive data.
    Open,
    /// A FIN has been sent or received; the stream is draining.
    HalfClosed,
    /// The stream has been cleanly closed (both sides have finished).
    Closed,
    /// The stream was abruptly reset (RST sent or received).
    Reset,
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamFrame
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed header length in bytes for [`StreamFrame`] wire encoding.
///
/// Layout: `stream_id(4) + sequence_num(8) + payload_len(4) + flags(1) + timestamp(8) = 25`.
const FRAME_HEADER_LEN: usize = 25;

/// A framed unit of data for one logical stream.
///
/// # Wire format (little-endian)
///
/// ```text
/// [stream_id: u32 LE][sequence_num: u64 LE][payload_len: u32 LE][flags: u8][timestamp: u64 LE][payload...]
/// ```
#[derive(Debug, Clone)]
pub struct StreamFrame {
    /// Which stream this frame belongs to.
    pub stream_id: StreamId,
    /// Monotonically increasing per-stream sequence number.
    pub sequence: u64,
    /// Payload bytes.
    pub data: Vec<u8>,
    /// Control flags: [`FLAG_FIN`], [`FLAG_RST`], [`FLAG_SYN`].
    pub flags: u8,
    /// Wall-clock timestamp at the time the frame was created (caller-supplied).
    pub timestamp: u64,
}

impl StreamFrame {
    // ── Legacy flag helpers (raw u8 constants) ─────────────────────────────

    /// Returns `true` if the FIN flag is set (legacy raw-u8 form).
    #[inline]
    pub fn is_fin(&self) -> bool {
        self.flags & FLAG_FIN != 0
    }

    /// Returns `true` if the RST flag is set (legacy raw-u8 form).
    #[inline]
    pub fn is_rst(&self) -> bool {
        self.flags & FLAG_RST != 0
    }

    /// Returns `true` if the SYN flag is set (legacy raw-u8 form).
    #[inline]
    pub fn is_syn(&self) -> bool {
        self.flags & FLAG_SYN != 0
    }

    /// Returns `true` if this is a pure control frame (no payload bytes matter).
    #[inline]
    pub fn is_control(&self) -> bool {
        self.flags != 0
    }

    // ── Wire encoding / decoding ───────────────────────────────────────────

    /// Encode the frame into a `Vec<u8>` using the fixed wire format.
    ///
    /// Wire layout (all little-endian):
    /// `stream_id(4) + sequence_num(8) + payload_len(4) + flags(1) + timestamp(8) + payload`
    ///
    /// Total header: 25 bytes.
    pub fn encode(&self) -> Vec<u8> {
        let payload_len = self.data.len();
        let mut buf = Vec::with_capacity(FRAME_HEADER_LEN + payload_len);
        buf.extend_from_slice(&self.stream_id.0.to_le_bytes());
        buf.extend_from_slice(&self.sequence.to_le_bytes());
        buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
        buf.push(self.flags);
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Decode a frame from a byte slice produced by [`StreamFrame::encode`].
    ///
    /// # Errors
    ///
    /// Returns [`MuxError::FrameTooLarge`] if `data` is shorter than the
    /// minimum header size or shorter than the declared payload length.
    pub fn decode(data: &[u8]) -> Result<Self, MuxError> {
        if data.len() < FRAME_HEADER_LEN {
            return Err(MuxError::FrameTooLarge(data.len()));
        }

        let stream_id = u32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| MuxError::FrameTooLarge(data.len()))?,
        );
        let sequence = u64::from_le_bytes(
            data[4..12]
                .try_into()
                .map_err(|_| MuxError::FrameTooLarge(data.len()))?,
        );
        let payload_len = u32::from_le_bytes(
            data[12..16]
                .try_into()
                .map_err(|_| MuxError::FrameTooLarge(data.len()))?,
        ) as usize;
        let flags = data[16];
        let timestamp = u64::from_le_bytes(
            data[17..25]
                .try_into()
                .map_err(|_| MuxError::FrameTooLarge(data.len()))?,
        );

        let expected_total = FRAME_HEADER_LEN + payload_len;
        if data.len() < expected_total {
            return Err(MuxError::FrameTooLarge(data.len()));
        }

        let payload = data[FRAME_HEADER_LEN..FRAME_HEADER_LEN + payload_len].to_vec();

        Ok(Self {
            stream_id: StreamId(stream_id),
            sequence,
            data: payload,
            flags,
            timestamp,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamInfo — read-only statistics snapshot for a single stream
// ─────────────────────────────────────────────────────────────────────────────

/// Read-only snapshot of per-stream statistics.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Stream identifier.
    pub id: StreamId,
    /// Current lifecycle state.
    pub state: StreamState,
    /// Total payload bytes enqueued for sending.
    pub bytes_sent: u64,
    /// Total payload bytes received.
    pub bytes_received: u64,
    /// Total frames enqueued for sending (including control frames).
    pub frames_sent: u64,
    /// Total frames received.
    pub frames_received: u64,
    /// Timestamp (µs) when the stream was opened.
    pub opened_at: u64,
    /// Timestamp (µs) of the last send or receive activity.
    pub last_activity: u64,
    /// Raw priority byte supplied at stream creation.
    pub priority: u8,
}

// ─────────────────────────────────────────────────────────────────────────────
// LogicalStream — internal mutable state for a single stream
// ─────────────────────────────────────────────────────────────────────────────

/// State and bookkeeping for a single logical stream.
#[derive(Debug, Clone)]
pub struct LogicalStream {
    /// Identifier of this stream.
    pub id: StreamId,
    /// Priority used for scheduling send frames.
    pub priority: StreamPriority,
    /// Current lifecycle state.
    pub state: StreamState,
    /// Remaining send window (bytes that may still be enqueued for sending).
    pub send_window: u32,
    /// Remaining receive window (bytes we are willing to accept from remote).
    pub recv_window: u32,
    /// Next sequence number to use when enqueuing a send frame.
    pub send_seq: u64,
    /// Expected sequence number for the next incoming frame.
    pub recv_seq: u64,
    /// Cumulative bytes enqueued for sending (payload only).
    pub bytes_sent: u64,
    /// Cumulative bytes received (payload only).
    pub bytes_received: u64,
    /// Cumulative frames enqueued for sending (including control frames).
    pub frames_sent: u64,
    /// Cumulative frames received.
    pub frames_received: u64,
    /// Timestamp (µs) when the stream was opened.
    pub opened_at: u64,
    /// Timestamp (µs) of the last activity (send or receive).
    pub last_activity: u64,
    /// Raw priority byte (0-255) supplied at stream creation.
    pub priority_raw: u8,
    /// Bounded receive buffer (payloads in arrival order).
    recv_buffer: VecDeque<Vec<u8>>,
    /// Maximum number of payloads buffered on the receive side.
    recv_buffer_max: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiplexerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`StreamMultiplexer`].
#[derive(Debug, Clone)]
pub struct MultiplexerConfig {
    /// Maximum number of concurrently open streams.
    pub max_streams: usize,
    /// Initial send/receive window size for each new stream (bytes).
    pub default_window_size: u32,
    /// Maximum payload bytes per frame when fragmenting.
    pub max_frame_size: usize,
    /// When `false`, all frames are enqueued as equal priority (FIFO).
    pub enable_priority: bool,
    /// Maximum number of payload entries held in the per-stream receive buffer.
    pub recv_buffer_capacity: usize,
    /// Maximum number of payload entries held in the per-stream send buffer
    /// (currently informational; actual cap is `send_window`).
    pub send_buffer_capacity: usize,
    /// Idle timeout in microseconds.  Streams inactive longer than this are
    /// eligible for expiry via [`StreamMultiplexer::expire_idle`].
    pub idle_timeout_us: u64,
}

impl Default for MultiplexerConfig {
    fn default() -> Self {
        Self {
            max_streams: 256,
            default_window_size: 65_536,
            max_frame_size: 16_384,
            enable_priority: true,
            recv_buffer_capacity: 256,
            send_buffer_capacity: 256,
            idle_timeout_us: 30_000_000, // 30 seconds
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiplexerStats — richer statistics snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Rich statistics snapshot for the multiplexer (task-spec version).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiplexerStats {
    /// Number of streams currently in the [`StreamState::Open`] or
    /// [`StreamState::Opening`] state.
    pub active_streams: usize,
    /// Total streams ever opened (monotonically increasing).
    pub total_streams_opened: u64,
    /// Total frames ever dequeued and handed to the caller.
    pub total_frames_sent: u64,
    /// Total frames processed via [`StreamMultiplexer::receive`].
    pub total_frames_received: u64,
    /// Total payload bytes enqueued for sending across all streams.
    pub total_bytes_sent: u64,
    /// Total payload bytes received across all streams.
    pub total_bytes_received: u64,
    /// Frames that were dropped due to a full receive buffer.
    pub dropped_frames: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PrioritizedFrame (internal heap element)
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper that gives [`StreamFrame`] a total ordering for the
/// [`BinaryHeap`]-based priority send queue.
///
/// Ordering rule: higher `priority_weight` first; ties broken by lower
/// enqueue `sequence` first (FIFO within a priority tier).
#[derive(Debug)]
struct PrioritizedFrame {
    /// Effective weight of the stream's priority at enqueue time.
    priority_weight: u32,
    /// Global monotonic enqueue counter used for FIFO within a priority tier.
    sequence: u64,
    /// The actual frame payload.
    frame: StreamFrame,
}

impl PartialEq for PrioritizedFrame {
    fn eq(&self, other: &Self) -> bool {
        self.priority_weight == other.priority_weight && self.sequence == other.sequence
    }
}

impl Eq for PrioritizedFrame {}

impl PartialOrd for PrioritizedFrame {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedFrame {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: higher weight wins (max-heap).
        self.priority_weight
            .cmp(&other.priority_weight)
            // Tie-break: lower enqueue sequence wins (FIFO).
            .then_with(|| Reverse(self.sequence).cmp(&Reverse(other.sequence)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MuxEvent
// ─────────────────────────────────────────────────────────────────────────────

/// Events emitted by [`StreamMultiplexer::receive`] and other operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxEvent {
    /// A new stream was opened (SYN received from remote).
    StreamOpened(StreamId),
    /// A stream was gracefully closed (FIN received).
    StreamClosed(StreamId),
    /// A stream was forcefully reset (RST received).
    StreamReset(StreamId),
    /// A data frame was received on the given stream.
    FrameReceived(StreamId),
    /// The receive buffer for the stream is full; a frame was dropped.
    SendBufferFull(StreamId),
    /// An idle stream was expired by [`StreamMultiplexer::expire_idle`].
    IdleStreamExpired(StreamId),
}

// ─────────────────────────────────────────────────────────────────────────────
// MuxError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`StreamMultiplexer`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxError {
    /// All stream slots are occupied.
    MaxStreamsReached,
    /// No stream with the given id exists.
    StreamNotFound(u32),
    /// The stream exists but is not in [`StreamState::Open`].
    StreamNotOpen(u32),
    /// The send window would be exceeded by the requested send.
    WindowExhausted(u32),
    /// The incoming frame carries an out-of-order sequence number.
    SequenceError {
        /// The sequence number we expected.
        expected: u64,
        /// The sequence number carried by the frame.
        got: u64,
    },
    /// The encoded frame is shorter than the minimum header length.
    FrameTooLarge(usize),
    /// The receive (or send) buffer for the stream is full.
    BufferOverflow(u32),
    /// A stream with the given id was already open when a SYN arrived.
    StreamAlreadyOpen(u32),
}

impl fmt::Display for MuxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxStreamsReached => write!(f, "maximum number of streams reached"),
            Self::StreamNotFound(id) => write!(f, "stream {id} not found"),
            Self::StreamNotOpen(id) => write!(f, "stream {id} is not open"),
            Self::WindowExhausted(id) => write!(f, "send window exhausted on stream {id}"),
            Self::SequenceError { expected, got } => {
                write!(f, "sequence error: expected {expected}, got {got}")
            }
            Self::FrameTooLarge(len) => write!(f, "frame data too short to decode ({len} bytes)"),
            Self::BufferOverflow(id) => write!(f, "buffer overflow on stream {id}"),
            Self::StreamAlreadyOpen(id) => write!(f, "stream {id} is already open"),
        }
    }
}

impl std::error::Error for MuxError {}

// ─────────────────────────────────────────────────────────────────────────────
// MuxStats — original compact statistics snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Compact snapshot of [`StreamMultiplexer`] statistics (original form).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuxStats {
    /// Total streams ever opened (including currently open ones).
    pub total_streams: usize,
    /// Number of streams currently in the [`StreamState::Open`] state.
    pub open_streams: usize,
    /// Frames currently waiting in the send queue.
    pub queued_frames: usize,
    /// Total frames that have been dequeued and handed to the caller.
    pub total_frames_sent: u64,
    /// Total frames that have been processed via [`StreamMultiplexer::receive`].
    pub total_frames_received: u64,
    /// Total payload bytes that have been enqueued for sending across all streams.
    pub total_bytes_sent: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamMultiplexer
// ─────────────────────────────────────────────────────────────────────────────

/// Multiplexes multiple logical streams over a single connection with
/// flow control and priority-based scheduling.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::stream_multiplexer::{
///     MultiplexerConfig, StreamMultiplexer, StreamPriority,
/// };
///
/// let config = MultiplexerConfig::default();
/// let mut mux = StreamMultiplexer::new(config);
///
/// let sid = mux.open_stream(StreamPriority::Normal).expect("open stream");
/// let bytes = mux.send(sid, b"hello world".to_vec(), 0).expect("send");
/// assert_eq!(bytes, 11);
///
/// // Drain all enqueued frames (SYN + data)
/// let frames = mux.dequeue_frames(10);
/// assert_eq!(frames.len(), 2);
/// ```
pub struct StreamMultiplexer {
    /// Effective configuration for this multiplexer instance.
    pub config: MultiplexerConfig,
    /// All known streams, keyed by their id.
    streams: HashMap<StreamId, LogicalStream>,
    /// Priority send queue.
    send_queue: BinaryHeap<PrioritizedFrame>,
    /// Monotonically increasing stream id allocator.
    next_stream_id: u32,
    /// Global enqueue counter used for FIFO ordering within a priority tier.
    enqueue_seq: u64,
    /// Total frames dequeued (sent to the caller).
    total_frames_sent: u64,
    /// Total frames processed via `receive`.
    total_frames_received: u64,
    /// Total streams ever opened.
    total_streams_opened: u64,
    /// Frames dropped due to a full receive buffer.
    dropped_frames: u64,
}

impl StreamMultiplexer {
    // ─────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────

    /// Create a new multiplexer with the given configuration.
    pub fn new(config: MultiplexerConfig) -> Self {
        Self {
            streams: HashMap::with_capacity(config.max_streams.min(256)),
            send_queue: BinaryHeap::new(),
            next_stream_id: 1,
            enqueue_seq: 0,
            total_frames_sent: 0,
            total_frames_received: 0,
            total_streams_opened: 0,
            dropped_frames: 0,
            config,
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Stream lifecycle
    // ─────────────────────────────────────────────────────────────

    /// Open a new logical stream with the specified priority.
    ///
    /// A SYN frame (flags = [`FLAG_SYN`]) is immediately enqueued.
    ///
    /// # Errors
    ///
    /// Returns [`MuxError::MaxStreamsReached`] if [`MultiplexerConfig::max_streams`]
    /// streams are already open.
    pub fn open_stream(&mut self, priority: StreamPriority) -> Result<StreamId, MuxError> {
        if self.streams.len() >= self.config.max_streams {
            return Err(MuxError::MaxStreamsReached);
        }

        let id = StreamId(self.next_stream_id);
        self.next_stream_id = self.next_stream_id.wrapping_add(1);

        let recv_cap = self.config.recv_buffer_capacity;
        let stream = LogicalStream {
            id,
            priority,
            state: StreamState::Open,
            send_window: self.config.default_window_size,
            recv_window: self.config.default_window_size,
            send_seq: 1, // 0 is consumed by the SYN frame below
            recv_seq: 0,
            bytes_sent: 0,
            bytes_received: 0,
            frames_sent: 1, // SYN frame counted here
            frames_received: 0,
            opened_at: 0,
            last_activity: 0,
            priority_raw: priority.weight() as u8,
            recv_buffer: VecDeque::with_capacity(recv_cap.min(64)),
            recv_buffer_max: recv_cap,
        };
        self.streams.insert(id, stream);
        self.total_streams_opened = self.total_streams_opened.wrapping_add(1);

        // Enqueue SYN frame with sequence 0.
        self.enqueue_control(id, FLAG_SYN, 0, priority, 0);

        Ok(id)
    }

    /// Open a new logical stream with a raw priority byte (0-255).
    ///
    /// The raw byte is mapped to the nearest [`StreamPriority`] tier via
    /// [`priority_from_u8`].  The raw value is preserved in [`LogicalStream::priority_raw`].
    ///
    /// # Errors
    ///
    /// Returns [`MuxError::MaxStreamsReached`] if the maximum stream count is exceeded.
    pub fn open_stream_with_priority(&mut self, priority_raw: u8) -> Result<StreamId, MuxError> {
        if self.streams.len() >= self.config.max_streams {
            return Err(MuxError::MaxStreamsReached);
        }

        let priority = priority_from_u8(priority_raw);
        let id = StreamId(self.next_stream_id);
        self.next_stream_id = self.next_stream_id.wrapping_add(1);

        let recv_cap = self.config.recv_buffer_capacity;
        let stream = LogicalStream {
            id,
            priority,
            state: StreamState::Open,
            send_window: self.config.default_window_size,
            recv_window: self.config.default_window_size,
            send_seq: 1,
            recv_seq: 0,
            bytes_sent: 0,
            bytes_received: 0,
            frames_sent: 1,
            frames_received: 0,
            opened_at: 0,
            last_activity: 0,
            priority_raw,
            recv_buffer: VecDeque::with_capacity(recv_cap.min(64)),
            recv_buffer_max: recv_cap,
        };
        self.streams.insert(id, stream);
        self.total_streams_opened = self.total_streams_opened.wrapping_add(1);

        self.enqueue_control(id, FLAG_SYN, 0, priority, 0);

        Ok(id)
    }

    /// Send a FIN frame and transition the stream to [`StreamState::HalfClosed`].
    ///
    /// # Errors
    ///
    /// - [`MuxError::StreamNotFound`] if the stream does not exist.
    /// - [`MuxError::StreamNotOpen`] if the stream is already Closed or Reset.
    pub fn close_stream(&mut self, stream_id: StreamId) -> Result<(), MuxError> {
        let stream = self
            .streams
            .get_mut(&stream_id)
            .ok_or(MuxError::StreamNotFound(stream_id.0))?;

        match stream.state {
            StreamState::Closed | StreamState::Reset => {
                return Err(MuxError::StreamNotOpen(stream_id.0));
            }
            _ => {}
        }

        let seq = stream.send_seq;
        stream.send_seq = stream.send_seq.wrapping_add(1);
        stream.frames_sent = stream.frames_sent.wrapping_add(1);
        let priority = stream.priority;
        stream.state = StreamState::HalfClosed;

        self.enqueue_control(stream_id, FLAG_FIN, seq, priority, 0);
        Ok(())
    }

    /// Send an RST frame and transition the stream to [`StreamState::Reset`].
    ///
    /// # Errors
    ///
    /// - [`MuxError::StreamNotFound`] if the stream does not exist.
    pub fn reset_stream(&mut self, stream_id: StreamId) -> Result<(), MuxError> {
        let stream = self
            .streams
            .get_mut(&stream_id)
            .ok_or(MuxError::StreamNotFound(stream_id.0))?;

        let seq = stream.send_seq;
        stream.send_seq = stream.send_seq.wrapping_add(1);
        stream.frames_sent = stream.frames_sent.wrapping_add(1);
        let priority = stream.priority;
        stream.state = StreamState::Reset;

        self.enqueue_control(stream_id, FLAG_RST, seq, priority, 0);
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────
    // Data path
    // ─────────────────────────────────────────────────────────────

    /// Fragment `data` into frames of at most [`MultiplexerConfig::max_frame_size`]
    /// bytes and enqueue them on the send queue.
    ///
    /// Returns the total number of bytes enqueued.
    ///
    /// # Errors
    ///
    /// - [`MuxError::StreamNotFound`] if the stream does not exist.
    /// - [`MuxError::StreamNotOpen`] if the stream is not [`StreamState::Open`].
    /// - [`MuxError::WindowExhausted`] if `data.len()` exceeds the remaining send window.
    pub fn send(
        &mut self,
        stream_id: StreamId,
        data: Vec<u8>,
        now: u64,
    ) -> Result<usize, MuxError> {
        // Validate before mutating.
        {
            let stream = self
                .streams
                .get(&stream_id)
                .ok_or(MuxError::StreamNotFound(stream_id.0))?;

            if stream.state != StreamState::Open {
                return Err(MuxError::StreamNotOpen(stream_id.0));
            }

            if data.len() as u64 > stream.send_window as u64 {
                return Err(MuxError::WindowExhausted(stream_id.0));
            }
        }

        let total = data.len();

        // Fragment and enqueue.
        let chunk_size = self.config.max_frame_size.max(1);
        let mut offset = 0usize;

        while offset < data.len() {
            let end = (offset + chunk_size).min(data.len());
            let chunk = data[offset..end].to_vec();
            let chunk_len = chunk.len();

            let stream = self
                .streams
                .get_mut(&stream_id)
                .ok_or(MuxError::StreamNotFound(stream_id.0))?;

            let seq = stream.send_seq;
            stream.send_seq = stream.send_seq.wrapping_add(1);
            stream.bytes_sent = stream.bytes_sent.saturating_add(chunk_len as u64);
            stream.frames_sent = stream.frames_sent.wrapping_add(1);
            // Deduct from send window immediately upon enqueue.
            stream.send_window = stream.send_window.saturating_sub(chunk_len as u32);
            stream.last_activity = now;
            let priority = stream.priority;
            let weight = if self.config.enable_priority {
                priority.weight()
            } else {
                0
            };

            let enqueue_seq = self.enqueue_seq;
            self.enqueue_seq = self.enqueue_seq.wrapping_add(1);

            self.send_queue.push(PrioritizedFrame {
                priority_weight: weight,
                sequence: enqueue_seq,
                frame: StreamFrame {
                    stream_id,
                    sequence: seq,
                    data: chunk,
                    flags: 0,
                    timestamp: now,
                },
            });

            offset = end;
        }

        Ok(total)
    }

    /// Process an incoming frame from the remote peer.
    ///
    /// Returns a list of [`MuxEvent`]s describing what happened:
    /// - SYN → [`MuxEvent::StreamOpened`] (opens the stream if not present)
    /// - DATA → [`MuxEvent::FrameReceived`] (payload buffered in recv buffer)
    /// - FIN → [`MuxEvent::StreamClosed`]
    /// - RST → [`MuxEvent::StreamReset`]
    /// - Recv buffer full → [`MuxEvent::SendBufferFull`] (frame dropped, counted in `dropped_frames`)
    ///
    /// # Errors
    ///
    /// - [`MuxError::StreamNotFound`] if the stream is unknown and frame is not SYN.
    /// - [`MuxError::SequenceError`] if the frame arrives out of order.
    pub fn receive_events(
        &mut self,
        frame: StreamFrame,
        now: u64,
    ) -> Result<Vec<MuxEvent>, MuxError> {
        let mut events = Vec::new();

        // SYN: open a new stream if it doesn't exist yet.
        if frame.is_syn() {
            if self.streams.contains_key(&frame.stream_id) {
                return Err(MuxError::StreamAlreadyOpen(frame.stream_id.0));
            }

            if self.streams.len() >= self.config.max_streams {
                return Err(MuxError::MaxStreamsReached);
            }

            let recv_cap = self.config.recv_buffer_capacity;
            let stream = LogicalStream {
                id: frame.stream_id,
                priority: StreamPriority::Normal,
                state: StreamState::Open,
                send_window: self.config.default_window_size,
                recv_window: self.config.default_window_size,
                send_seq: 0,
                recv_seq: 1, // SYN consumed sequence 0
                bytes_sent: 0,
                bytes_received: 0,
                frames_sent: 0,
                frames_received: 1,
                opened_at: now,
                last_activity: now,
                priority_raw: StreamPriority::Normal.weight() as u8,
                recv_buffer: VecDeque::with_capacity(recv_cap.min(64)),
                recv_buffer_max: recv_cap,
            };
            self.streams.insert(frame.stream_id, stream);
            self.total_streams_opened = self.total_streams_opened.wrapping_add(1);
            self.total_frames_received = self.total_frames_received.wrapping_add(1);
            events.push(MuxEvent::StreamOpened(frame.stream_id));
            return Ok(events);
        }

        let stream = self
            .streams
            .get_mut(&frame.stream_id)
            .ok_or(MuxError::StreamNotFound(frame.stream_id.0))?;

        // RST takes immediate effect regardless of sequence.
        if frame.is_rst() {
            stream.state = StreamState::Reset;
            stream.last_activity = now;
            stream.frames_received = stream.frames_received.wrapping_add(1);
            self.total_frames_received = self.total_frames_received.wrapping_add(1);
            events.push(MuxEvent::StreamReset(frame.stream_id));
            return Ok(events);
        }

        // Sequence check.
        if frame.sequence != stream.recv_seq {
            return Err(MuxError::SequenceError {
                expected: stream.recv_seq,
                got: frame.sequence,
            });
        }

        stream.recv_seq = stream.recv_seq.wrapping_add(1);
        let data_len = frame.data.len();
        stream.frames_received = stream.frames_received.wrapping_add(1);
        stream.last_activity = now;
        self.total_frames_received = self.total_frames_received.wrapping_add(1);

        if frame.is_fin() {
            stream.bytes_received = stream.bytes_received.saturating_add(data_len as u64);
            stream.state = if stream.state == StreamState::HalfClosed {
                StreamState::Closed
            } else {
                StreamState::HalfClosed
            };
            events.push(MuxEvent::StreamClosed(frame.stream_id));
        } else if !frame.data.is_empty() {
            // DATA frame: buffer the payload.
            stream.bytes_received = stream.bytes_received.saturating_add(data_len as u64);
            if stream.recv_buffer.len() >= stream.recv_buffer_max {
                self.dropped_frames = self.dropped_frames.wrapping_add(1);
                events.push(MuxEvent::SendBufferFull(frame.stream_id));
            } else {
                stream.recv_buffer.push_back(frame.data);
                events.push(MuxEvent::FrameReceived(frame.stream_id));
            }
        } else {
            events.push(MuxEvent::FrameReceived(frame.stream_id));
        }

        Ok(events)
    }

    /// Process an incoming frame from the remote peer (original API).
    ///
    /// Handles RST, FIN, and data frames.  Returns the payload bytes.
    ///
    /// # Errors
    ///
    /// - [`MuxError::StreamNotFound`] if the stream is unknown.
    /// - [`MuxError::SequenceError`] if the frame arrives out of order.
    pub fn receive(&mut self, frame: StreamFrame, _now: u64) -> Result<Vec<u8>, MuxError> {
        let stream = self
            .streams
            .get_mut(&frame.stream_id)
            .ok_or(MuxError::StreamNotFound(frame.stream_id.0))?;

        // RST takes immediate effect regardless of sequence.
        if frame.is_rst() {
            stream.state = StreamState::Reset;
            self.total_frames_received = self.total_frames_received.wrapping_add(1);
            return Ok(Vec::new());
        }

        // Sequence check (SYN frames reset the counter expectation).
        if !frame.is_syn() && frame.sequence != stream.recv_seq {
            return Err(MuxError::SequenceError {
                expected: stream.recv_seq,
                got: frame.sequence,
            });
        }

        stream.recv_seq = stream.recv_seq.wrapping_add(1);
        let data_len = frame.data.len();
        stream.bytes_received = stream.bytes_received.saturating_add(data_len as u64);

        if frame.is_fin() {
            stream.state = StreamState::HalfClosed;
        }

        self.total_frames_received = self.total_frames_received.wrapping_add(1);
        Ok(frame.data)
    }

    // ─────────────────────────────────────────────────────────────
    // Receive buffer drain
    // ─────────────────────────────────────────────────────────────

    /// Drain all buffered receive payloads for `stream_id`.
    ///
    /// Returns a `Vec` of payload byte vectors in arrival order.
    /// The receive buffer is emptied after this call.
    ///
    /// # Errors
    ///
    /// Returns [`MuxError::StreamNotFound`] if the stream does not exist.
    pub fn drain_recv_buffer(&mut self, stream_id: StreamId) -> Result<Vec<Vec<u8>>, MuxError> {
        let stream = self
            .streams
            .get_mut(&stream_id)
            .ok_or(MuxError::StreamNotFound(stream_id.0))?;

        let payloads: Vec<Vec<u8>> = stream.recv_buffer.drain(..).collect();
        Ok(payloads)
    }

    // ─────────────────────────────────────────────────────────────
    // Dequeue
    // ─────────────────────────────────────────────────────────────

    /// Pop the highest-priority frame from the send queue.
    ///
    /// Updates `total_frames_sent`.  For data frames (no control flags),
    /// the stream's `send_window` has already been decremented in `send`;
    /// no further deduction is needed here.
    pub fn dequeue_frame(&mut self) -> Option<StreamFrame> {
        let pf = self.send_queue.pop()?;
        self.total_frames_sent = self.total_frames_sent.wrapping_add(1);
        Some(pf.frame)
    }

    /// Pop up to `n` frames from the send queue in priority order.
    pub fn dequeue_frames(&mut self, n: usize) -> Vec<StreamFrame> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            match self.dequeue_frame() {
                Some(f) => result.push(f),
                None => break,
            }
        }
        result
    }

    /// Drain **all** pending outbound frames across all streams in priority order.
    ///
    /// Equivalent to calling `dequeue_frames` with `usize::MAX`, but more
    /// efficient since it drains the heap directly.
    pub fn drain_send_queue(&mut self) -> Vec<StreamFrame> {
        let mut result = Vec::with_capacity(self.send_queue.len());
        while let Some(pf) = self.send_queue.pop() {
            self.total_frames_sent = self.total_frames_sent.wrapping_add(1);
            result.push(pf.frame);
        }
        result
    }

    // ─────────────────────────────────────────────────────────────
    // Idle expiry
    // ─────────────────────────────────────────────────────────────

    /// Expire streams that have been idle longer than
    /// [`MultiplexerConfig::idle_timeout_us`].
    ///
    /// A stream is considered idle if
    /// `last_activity + idle_timeout_us < current_ts`.
    ///
    /// Expired streams are transitioned to [`StreamState::Closed`] and a
    /// [`MuxEvent::IdleStreamExpired`] event is emitted for each one.
    pub fn expire_idle(&mut self, current_ts: u64) -> Vec<MuxEvent> {
        let timeout = self.config.idle_timeout_us;
        let mut expired = Vec::new();

        for (id, stream) in self.streams.iter_mut() {
            if matches!(
                stream.state,
                StreamState::Open | StreamState::Opening | StreamState::HalfClosed
            ) && stream.last_activity.saturating_add(timeout) < current_ts
            {
                stream.state = StreamState::Closed;
                expired.push(*id);
            }
        }

        expired
            .into_iter()
            .map(MuxEvent::IdleStreamExpired)
            .collect()
    }

    // ─────────────────────────────────────────────────────────────
    // Flow control
    // ─────────────────────────────────────────────────────────────

    /// Add `increment` bytes to the send window for `stream_id`.
    ///
    /// Returns `true` if the stream exists and the window was updated,
    /// `false` otherwise.
    pub fn update_window(&mut self, stream_id: StreamId, increment: u32) -> bool {
        if let Some(stream) = self.streams.get_mut(&stream_id) {
            stream.send_window = stream.send_window.saturating_add(increment);
            true
        } else {
            false
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Introspection
    // ─────────────────────────────────────────────────────────────

    /// Return a reference to the [`LogicalStream`] for the given id, or `None`.
    pub fn stream_state(&self, stream_id: StreamId) -> Option<&LogicalStream> {
        self.streams.get(&stream_id)
    }

    /// Return a read-only [`StreamInfo`] snapshot for the given stream.
    ///
    /// # Errors
    ///
    /// Returns [`MuxError::StreamNotFound`] if the stream does not exist.
    pub fn stream_info(&self, stream_id: StreamId) -> Result<StreamInfo, MuxError> {
        let s = self
            .streams
            .get(&stream_id)
            .ok_or(MuxError::StreamNotFound(stream_id.0))?;

        Ok(StreamInfo {
            id: s.id,
            state: s.state,
            bytes_sent: s.bytes_sent,
            bytes_received: s.bytes_received,
            frames_sent: s.frames_sent,
            frames_received: s.frames_received,
            opened_at: s.opened_at,
            last_activity: s.last_activity,
            priority: s.priority_raw,
        })
    }

    /// Return the ids of all currently active (Open or Opening) streams.
    pub fn active_streams(&self) -> Vec<StreamId> {
        self.streams
            .values()
            .filter(|s| matches!(s.state, StreamState::Open | StreamState::Opening))
            .map(|s| s.id)
            .collect()
    }

    /// Number of streams currently in [`StreamState::Open`].
    pub fn open_stream_count(&self) -> usize {
        self.streams
            .values()
            .filter(|s| s.state == StreamState::Open)
            .count()
    }

    /// Return the original compact statistics snapshot ([`MuxStats`]).
    pub fn stats(&self) -> MuxStats {
        let total_bytes_sent = self
            .streams
            .values()
            .map(|s| s.bytes_sent)
            .fold(0u64, |acc, b| acc.saturating_add(b));

        MuxStats {
            total_streams: self.streams.len(),
            open_streams: self.open_stream_count(),
            queued_frames: self.send_queue.len(),
            total_frames_sent: self.total_frames_sent,
            total_frames_received: self.total_frames_received,
            total_bytes_sent,
        }
    }

    /// Return the richer [`MultiplexerStats`] snapshot.
    pub fn multiplexer_stats(&self) -> MultiplexerStats {
        let (total_bytes_sent, total_bytes_received) =
            self.streams.values().fold((0u64, 0u64), |(s, r), stream| {
                (
                    s.saturating_add(stream.bytes_sent),
                    r.saturating_add(stream.bytes_received),
                )
            });

        let active = self
            .streams
            .values()
            .filter(|s| matches!(s.state, StreamState::Open | StreamState::Opening))
            .count();

        MultiplexerStats {
            active_streams: active,
            total_streams_opened: self.total_streams_opened,
            total_frames_sent: self.total_frames_sent,
            total_frames_received: self.total_frames_received,
            total_bytes_sent,
            total_bytes_received,
            dropped_frames: self.dropped_frames,
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────

    /// Enqueue a control frame (SYN / FIN / RST) without touching the send window.
    fn enqueue_control(
        &mut self,
        stream_id: StreamId,
        flags: u8,
        sequence: u64,
        priority: StreamPriority,
        now: u64,
    ) {
        let weight = if self.config.enable_priority {
            priority.weight()
        } else {
            0
        };
        let enqueue_seq = self.enqueue_seq;
        self.enqueue_seq = self.enqueue_seq.wrapping_add(1);

        self.send_queue.push(PrioritizedFrame {
            priority_weight: weight,
            sequence: enqueue_seq,
            frame: StreamFrame {
                stream_id,
                sequence,
                data: Vec::new(),
                flags,
                timestamp: now,
            },
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PRNG helper (no external rand dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG — seedable, dependency-free random number source for tests.
///
/// # Example
/// ```ignore
/// let mut state = 12345u64;
/// let r = xorshift64(&mut state);
/// ```
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (split to stream_multiplexer_tests.rs to keep this file < 2000 lines)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "stream_multiplexer_tests.rs"]
mod tests;
