//! Tests for [`crate::stream_multiplexer`].
//!
//! Split into a dedicated file to keep the main module under 2000 lines.

use crate::stream_multiplexer::{
    priority_from_u8, xorshift64, FrameFlags, LogicalStream, MultiplexerConfig, MultiplexerStats,
    MuxError, MuxEvent, MuxStats, StreamFrame, StreamId, StreamInfo, StreamMultiplexer,
    StreamPriority, StreamState, FLAG_FIN, FLAG_RST, FLAG_SYN,
};

fn default_mux() -> StreamMultiplexer {
    StreamMultiplexer::new(MultiplexerConfig::default())
}

fn mux_with_max(max: usize) -> StreamMultiplexer {
    StreamMultiplexer::new(MultiplexerConfig {
        max_streams: max,
        ..MultiplexerConfig::default()
    })
}

// ── T01: open_stream returns a valid StreamId ──────────────────────────
#[test]
fn t01_open_stream_returns_id() {
    let mut mux = default_mux();
    let id = mux.open_stream(StreamPriority::Normal).expect("open");
    assert_eq!(id, StreamId(1));
}

// ── T02: sequential opens give incrementing ids ────────────────────────
#[test]
fn t02_sequential_ids() {
    let mut mux = default_mux();
    let a = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let b = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    assert_eq!(b.0, a.0 + 1);
}

// ── T03: open_stream enqueues a SYN frame ─────────────────────────────
#[test]
fn t03_open_enqueues_syn() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = mux.dequeue_frame().expect("frame");
    assert_eq!(frame.stream_id, id);
    assert!(frame.is_syn());
}

// ── T04: stream is Open after open_stream ─────────────────────────────
#[test]
fn t04_stream_starts_open() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::High)
        .expect("test: open_stream should succeed");
    let s = mux.stream_state(id).expect("state");
    assert_eq!(s.state, StreamState::Open);
}

// ── T05: max_streams limit ────────────────────────────────────────────
#[test]
fn t05_max_streams_limit() {
    let mut mux = mux_with_max(2);
    mux.open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let err = mux.open_stream(StreamPriority::Normal).unwrap_err();
    assert_eq!(err, MuxError::MaxStreamsReached);
}

// ── T06: send returns byte count ──────────────────────────────────────
#[test]
fn t06_send_returns_byte_count() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let n = mux
        .send(id, b"hello".to_vec(), 0)
        .expect("test: send should succeed");
    assert_eq!(n, 5);
}

// ── T07: send on nonexistent stream ───────────────────────────────────
#[test]
fn t07_send_missing_stream() {
    let mut mux = default_mux();
    let err = mux.send(StreamId(99), b"x".to_vec(), 0).unwrap_err();
    assert_eq!(err, MuxError::StreamNotFound(99));
}

// ── T08: send on closed stream ────────────────────────────────────────
#[test]
fn t08_send_closed_stream() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.close_stream(id)
        .expect("test: close_stream should succeed");
    let err = mux.send(id, b"x".to_vec(), 0).unwrap_err();
    assert_eq!(err, MuxError::StreamNotOpen(id.0));
}

// ── T09: send on reset stream ─────────────────────────────────────────
#[test]
fn t09_send_reset_stream() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.reset_stream(id)
        .expect("test: reset_stream should succeed");
    let err = mux.send(id, b"x".to_vec(), 0).unwrap_err();
    assert_eq!(err, MuxError::StreamNotOpen(id.0));
}

// ── T10: window exhaustion ────────────────────────────────────────────
#[test]
fn t10_window_exhausted() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        default_window_size: 4,
        ..MultiplexerConfig::default()
    });
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    // 5 bytes > 4-byte window
    let err = mux.send(id, b"12345".to_vec(), 0).unwrap_err();
    assert_eq!(err, MuxError::WindowExhausted(id.0));
}

// ── T11: send fragments large data ────────────────────────────────────
#[test]
fn t11_send_fragments() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        max_frame_size: 3,
        default_window_size: 65536,
        ..MultiplexerConfig::default()
    });
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let n = mux
        .send(id, b"abcdef".to_vec(), 0)
        .expect("test: send should succeed");
    assert_eq!(n, 6);
    // SYN + 2 data frames (abc, def)
    let frames = mux.dequeue_frames(10);
    let data_frames: Vec<_> = frames.iter().filter(|f| !f.is_control()).collect();
    assert_eq!(data_frames.len(), 2);
    assert_eq!(data_frames[0].data, b"abc");
    assert_eq!(data_frames[1].data, b"def");
}

// ── T12: dequeue_frame returns None when empty ────────────────────────
#[test]
fn t12_dequeue_empty() {
    let mut mux = default_mux();
    assert!(mux.dequeue_frame().is_none());
}

// ── T13: dequeue_frames with n=0 ─────────────────────────────────────
#[test]
fn t13_dequeue_frames_zero() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.send(id, b"data".to_vec(), 0)
        .expect("test: send should succeed");
    let frames = mux.dequeue_frames(0);
    assert!(frames.is_empty());
}

// ── T14: dequeue respects priority ────────────────────────────────────
#[test]
fn t14_priority_ordering() {
    let mut mux = default_mux();
    let lo = mux
        .open_stream(StreamPriority::Low)
        .expect("test: open_stream should succeed");
    let hi = mux
        .open_stream(StreamPriority::Critical)
        .expect("test: open_stream should succeed");

    // Drain SYN frames first.
    mux.dequeue_frames(2);

    mux.send(lo, b"low".to_vec(), 0)
        .expect("test: send should succeed");
    mux.send(hi, b"high".to_vec(), 0)
        .expect("test: send should succeed");

    let first = mux.dequeue_frame().expect("first");
    // Critical weight (8) > Low weight (1) → high-priority frame dequeued first.
    assert_eq!(first.stream_id, hi);
}

// ── T15: FIFO ordering within same priority ───────────────────────────
#[test]
fn t15_fifo_within_priority() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        max_frame_size: 3,
        default_window_size: 65536,
        ..MultiplexerConfig::default()
    });
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.dequeue_frame(); // drain SYN

    mux.send(id, b"AAABBB".to_vec(), 0)
        .expect("test: send should succeed");
    let f1 = mux.dequeue_frame().expect("test: frame should be dequeued");
    let f2 = mux.dequeue_frame().expect("test: frame should be dequeued");
    assert_eq!(f1.data, b"AAA");
    assert_eq!(f2.data, b"BBB");
}

// ── T16: close_stream enqueues FIN ────────────────────────────────────
#[test]
fn t16_close_enqueues_fin() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.dequeue_frame(); // drain SYN
    mux.close_stream(id)
        .expect("test: close_stream should succeed");
    let frame = mux.dequeue_frame().expect("fin");
    assert!(frame.is_fin());
    assert_eq!(frame.stream_id, id);
}

// ── T17: close_stream → HalfClosed ───────────────────────────────────
#[test]
fn t17_close_sets_half_closed() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.close_stream(id)
        .expect("test: close_stream should succeed");
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .state,
        StreamState::HalfClosed
    );
}

// ── T18: close_stream on unknown stream ───────────────────────────────
#[test]
fn t18_close_missing() {
    let mut mux = default_mux();
    let err = mux.close_stream(StreamId(42)).unwrap_err();
    assert_eq!(err, MuxError::StreamNotFound(42));
}

// ── T19: close_stream on Reset stream ────────────────────────────────
#[test]
fn t19_close_reset_stream() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.reset_stream(id)
        .expect("test: reset_stream should succeed");
    let err = mux.close_stream(id).unwrap_err();
    assert_eq!(err, MuxError::StreamNotOpen(id.0));
}

// ── T20: reset_stream enqueues RST ────────────────────────────────────
#[test]
fn t20_reset_enqueues_rst() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.dequeue_frame(); // drain SYN
    mux.reset_stream(id)
        .expect("test: reset_stream should succeed");
    let frame = mux.dequeue_frame().expect("rst");
    assert!(frame.is_rst());
}

// ── T21: reset_stream → Reset state ──────────────────────────────────
#[test]
fn t21_reset_sets_state() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.reset_stream(id)
        .expect("test: reset_stream should succeed");
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .state,
        StreamState::Reset
    );
}

// ── T22: reset_stream on unknown stream ───────────────────────────────
#[test]
fn t22_reset_missing() {
    let mut mux = default_mux();
    let err = mux.reset_stream(StreamId(7)).unwrap_err();
    assert_eq!(err, MuxError::StreamNotFound(7));
}

// ── T23: receive happy path ───────────────────────────────────────────
#[test]
fn t23_receive_data() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = StreamFrame {
        stream_id: id,
        sequence: 0,
        data: b"world".to_vec(),
        flags: 0,
        timestamp: 1,
    };
    let data = mux.receive(frame, 1).expect("test: receive should succeed");
    assert_eq!(data, b"world");
}

// ── T24: receive increments recv_seq ─────────────────────────────────
#[test]
fn t24_receive_increments_seq() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    for seq in 0..3u64 {
        let frame = StreamFrame {
            stream_id: id,
            sequence: seq,
            data: vec![0u8],
            flags: 0,
            timestamp: 0,
        };
        mux.receive(frame, 0).expect("test: receive should succeed");
    }
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .recv_seq,
        3
    );
}

// ── T25: receive out-of-order → SequenceError ─────────────────────────
#[test]
fn t25_receive_out_of_order() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = StreamFrame {
        stream_id: id,
        sequence: 5,
        data: b"bad".to_vec(),
        flags: 0,
        timestamp: 0,
    };
    let err = mux.receive(frame, 0).unwrap_err();
    assert!(matches!(
        err,
        MuxError::SequenceError {
            expected: 0,
            got: 5
        }
    ));
}

// ── T26: receive FIN → HalfClosed ─────────────────────────────────────
#[test]
fn t26_receive_fin() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = StreamFrame {
        stream_id: id,
        sequence: 0,
        data: Vec::new(),
        flags: FLAG_FIN,
        timestamp: 0,
    };
    mux.receive(frame, 0).expect("test: receive should succeed");
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .state,
        StreamState::HalfClosed
    );
}

// ── T27: receive RST → Reset ──────────────────────────────────────────
#[test]
fn t27_receive_rst() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = StreamFrame {
        stream_id: id,
        sequence: 99, // sequence is ignored for RST
        data: Vec::new(),
        flags: FLAG_RST,
        timestamp: 0,
    };
    mux.receive(frame, 0).expect("test: receive should succeed");
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .state,
        StreamState::Reset
    );
}

// ── T28: receive unknown stream ───────────────────────────────────────
#[test]
fn t28_receive_unknown_stream() {
    let mut mux = default_mux();
    let frame = StreamFrame {
        stream_id: StreamId(77),
        sequence: 0,
        data: b"x".to_vec(),
        flags: 0,
        timestamp: 0,
    };
    let err = mux.receive(frame, 0).unwrap_err();
    assert_eq!(err, MuxError::StreamNotFound(77));
}

// ── T29: update_window returns true for known stream ──────────────────
#[test]
fn t29_update_window_known() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let before = mux
        .stream_state(id)
        .expect("test: stream should exist")
        .send_window;
    let found = mux.update_window(id, 1024);
    assert!(found);
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .send_window,
        before + 1024
    );
}

// ── T30: update_window returns false for unknown stream ───────────────
#[test]
fn t30_update_window_unknown() {
    let mut mux = default_mux();
    assert!(!mux.update_window(StreamId(999), 100));
}

// ── T31: stats — queued_frames reflects queue size ────────────────────
#[test]
fn t31_stats_queued_frames() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.send(id, b"hello".to_vec(), 0)
        .expect("test: send should succeed");
    // 1 SYN + 1 data frame = 2 queued
    assert_eq!(mux.stats().queued_frames, 2);
}

// ── T32: stats — total_frames_sent increments ─────────────────────────
#[test]
fn t32_stats_frames_sent() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.send(id, b"x".to_vec(), 0)
        .expect("test: send should succeed");
    mux.dequeue_frame(); // SYN
    mux.dequeue_frame(); // data
    assert_eq!(mux.stats().total_frames_sent, 2);
}

// ── T33: stats — total_frames_received increments ─────────────────────
#[test]
fn t33_stats_frames_received() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = StreamFrame {
        stream_id: id,
        sequence: 0,
        data: b"abc".to_vec(),
        flags: 0,
        timestamp: 0,
    };
    mux.receive(frame, 0).expect("test: receive should succeed");
    assert_eq!(mux.stats().total_frames_received, 1);
}

// ── T34: stats — open_streams count ───────────────────────────────────
#[test]
fn t34_stats_open_streams() {
    let mut mux = default_mux();
    let a = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.close_stream(a)
        .expect("test: close_stream should succeed");
    let stats = mux.stats();
    assert_eq!(stats.open_streams, 1);
    assert_eq!(stats.total_streams, 2);
}

// ── T35: stats — total_bytes_sent ─────────────────────────────────────
#[test]
fn t35_stats_bytes_sent() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.send(id, b"hello world".to_vec(), 0)
        .expect("test: send should succeed");
    assert_eq!(mux.stats().total_bytes_sent, 11);
}

// ── T36: open_stream_count ────────────────────────────────────────────
#[test]
fn t36_open_stream_count() {
    let mut mux = default_mux();
    assert_eq!(mux.open_stream_count(), 0);
    let a = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let b = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    assert_eq!(mux.open_stream_count(), 2);
    mux.reset_stream(a)
        .expect("test: reset_stream should succeed");
    assert_eq!(mux.open_stream_count(), 1);
    mux.close_stream(b)
        .expect("test: close_stream should succeed");
    assert_eq!(mux.open_stream_count(), 0);
}

// ── T37: stream_state returns None for unknown ────────────────────────
#[test]
fn t37_stream_state_none() {
    let mux = default_mux();
    assert!(mux.stream_state(StreamId(0)).is_none());
}

// ── T38: priority disabled → FIFO across streams ──────────────────────
#[test]
fn t38_priority_disabled_fifo() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        enable_priority: false,
        ..MultiplexerConfig::default()
    });
    let lo = mux
        .open_stream(StreamPriority::Low)
        .expect("test: open_stream should succeed");
    let hi = mux
        .open_stream(StreamPriority::Critical)
        .expect("test: open_stream should succeed");
    // Drain SYN frames (lo SYN enqueued first → dequeued first)
    let s1 = mux.dequeue_frame().expect("test: frame should be dequeued");
    let s2 = mux.dequeue_frame().expect("test: frame should be dequeued");
    assert_eq!(s1.stream_id, lo); // lo was opened first
    assert_eq!(s2.stream_id, hi);

    mux.send(lo, b"low".to_vec(), 0)
        .expect("test: send should succeed");
    mux.send(hi, b"high".to_vec(), 0)
        .expect("test: send should succeed");
    // With priority disabled all weights = 0 → FIFO: lo data first
    let f1 = mux.dequeue_frame().expect("test: frame should be dequeued");
    assert_eq!(f1.stream_id, lo);
}

// ── T39: window is decremented on send ───────────────────────────────
#[test]
fn t39_send_decrements_window() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        default_window_size: 100,
        ..MultiplexerConfig::default()
    });
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.send(id, b"hello".to_vec(), 0)
        .expect("test: send should succeed"); // 5 bytes
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .send_window,
        95
    );
}

// ── T40: window update after window exhausted allows resend ───────────
#[test]
fn t40_window_update_allows_resend() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        default_window_size: 5,
        ..MultiplexerConfig::default()
    });
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.send(id, b"hello".to_vec(), 0)
        .expect("test: send should succeed"); // consumes window
                                              // Now window = 0 → next send fails
    assert!(matches!(
        mux.send(id, b"x".to_vec(), 0),
        Err(MuxError::WindowExhausted(_))
    ));
    // Update window
    mux.update_window(id, 10);
    // Now it succeeds
    assert_eq!(
        mux.send(id, b"x".to_vec(), 0)
            .expect("test: send should succeed"),
        1
    );
}

// ── T41: receive increments bytes_received ────────────────────────────
#[test]
fn t41_receive_increments_bytes() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = StreamFrame {
        stream_id: id,
        sequence: 0,
        data: b"testing".to_vec(),
        flags: 0,
        timestamp: 0,
    };
    mux.receive(frame, 0).expect("test: receive should succeed");
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .bytes_received,
        7
    );
}

// ── T42: multiple streams send independently ──────────────────────────
#[test]
fn t42_multiple_streams_independent() {
    let mut mux = default_mux();
    let a = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let b = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.send(a, b"from a".to_vec(), 0)
        .expect("test: send should succeed");
    mux.send(b, b"from b".to_vec(), 0)
        .expect("test: send should succeed");
    assert_eq!(
        mux.stream_state(a)
            .expect("test: stream should exist")
            .bytes_sent,
        6
    );
    assert_eq!(
        mux.stream_state(b)
            .expect("test: stream should exist")
            .bytes_sent,
        6
    );
}

// ── T43: MuxError Display ─────────────────────────────────────────────
#[test]
fn t43_error_display() {
    assert!(!format!("{}", MuxError::MaxStreamsReached).is_empty());
    assert!(!format!("{}", MuxError::StreamNotFound(1)).is_empty());
    assert!(!format!("{}", MuxError::StreamNotOpen(2)).is_empty());
    assert!(!format!("{}", MuxError::WindowExhausted(3)).is_empty());
    assert!(!format!(
        "{}",
        MuxError::SequenceError {
            expected: 0,
            got: 1
        }
    )
    .is_empty());
    assert!(!format!("{}", MuxError::FrameTooLarge(5)).is_empty());
    assert!(!format!("{}", MuxError::BufferOverflow(9)).is_empty());
    assert!(!format!("{}", MuxError::StreamAlreadyOpen(4)).is_empty());
}

// ── T44: StreamFrame flag helpers ────────────────────────────────────
#[test]
fn t44_frame_flag_helpers() {
    let fin = StreamFrame {
        stream_id: StreamId(1),
        sequence: 0,
        data: Vec::new(),
        flags: FLAG_FIN,
        timestamp: 0,
    };
    assert!(fin.is_fin());
    assert!(!fin.is_rst());
    assert!(!fin.is_syn());
    assert!(fin.is_control());

    let data = StreamFrame {
        stream_id: StreamId(1),
        sequence: 0,
        data: b"x".to_vec(),
        flags: 0,
        timestamp: 0,
    };
    assert!(!data.is_control());
}

// ── T45: StreamPriority weights ───────────────────────────────────────
#[test]
fn t45_priority_weights() {
    assert_eq!(StreamPriority::Critical.weight(), 8);
    assert_eq!(StreamPriority::High.weight(), 4);
    assert_eq!(StreamPriority::Normal.weight(), 2);
    assert_eq!(StreamPriority::Low.weight(), 1);
    assert_eq!(StreamPriority::Background.weight(), 0);
}

// ── T46: dequeue_frames caps at available ─────────────────────────────
#[test]
fn t46_dequeue_frames_capped() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.send(id, b"abc".to_vec(), 0)
        .expect("test: send should succeed");
    // 2 frames (SYN + data) but request 100
    let frames = mux.dequeue_frames(100);
    assert_eq!(frames.len(), 2);
}

// ── T47: SYN frame sequence is 0 ─────────────────────────────────────
#[test]
fn t47_syn_sequence_zero() {
    let mut mux = default_mux();
    mux.open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = mux.dequeue_frame().expect("test: frame should be dequeued");
    assert_eq!(frame.sequence, 0);
}

// ── T48: close HalfClosed stream is allowed ───────────────────────────
#[test]
fn t48_close_half_closed_ok() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.close_stream(id)
        .expect("test: close_stream should succeed"); // → HalfClosed
                                                      // Closing a HalfClosed stream again is allowed (sends another FIN)
    assert!(mux.close_stream(id).is_ok());
}

// ── T49: send empty data ──────────────────────────────────────────────
#[test]
fn t49_send_empty_data() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    // Empty send: 0 bytes, no frames enqueued beyond SYN
    let n = mux
        .send(id, Vec::new(), 0)
        .expect("test: send should succeed");
    assert_eq!(n, 0);
    // Only SYN should be in queue
    assert_eq!(mux.stats().queued_frames, 1);
}

// ── T50: MuxStats is Debug + Clone ───────────────────────────────────
#[test]
fn t50_mux_stats_traits() {
    let mux = default_mux();
    let stats: MuxStats = mux.stats();
    let _cloned = stats.clone();
    let _debug = format!("{:?}", stats);
}

// ── T51: LogicalStream is Debug + Clone ──────────────────────────────
#[test]
fn t51_logical_stream_traits() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let stream: &LogicalStream = mux.stream_state(id).expect("test: stream should exist");
    let _cloned = stream.clone();
    let _debug = format!("{:?}", stream);
}

// ── T52: send_seq advances per frame ─────────────────────────────────
#[test]
fn t52_send_seq_advances() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        max_frame_size: 1,
        default_window_size: 65536,
        ..MultiplexerConfig::default()
    });
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    // send_seq starts at 0 (SYN consumed 0)
    mux.send(id, b"ab".to_vec(), 0)
        .expect("test: send should succeed");
    // SYN used seq 0, data frames used seq 1 and 2
    let stream = mux.stream_state(id).expect("test: stream should exist");
    // send_seq should be at 3 (0=SYN, 1=a, 2=b)
    assert_eq!(stream.send_seq, 3);
}

// ── T53: receive RST ignores sequence number ──────────────────────────
#[test]
fn t53_receive_rst_ignores_seq() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    // Receive some data first to advance recv_seq
    for i in 0..3u64 {
        mux.receive(
            StreamFrame {
                stream_id: id,
                sequence: i,
                data: vec![1],
                flags: 0,
                timestamp: 0,
            },
            0,
        )
        .expect("test: receive should succeed");
    }
    // RST with wrong sequence — should still succeed
    let rst = StreamFrame {
        stream_id: id,
        sequence: 999,
        data: Vec::new(),
        flags: FLAG_RST,
        timestamp: 0,
    };
    assert!(mux.receive(rst, 0).is_ok());
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .state,
        StreamState::Reset
    );
}

// ── T54: StreamFrame encode/decode round-trip ─────────────────────────
#[test]
fn t54_frame_encode_decode_roundtrip() {
    let original = StreamFrame {
        stream_id: StreamId(42),
        sequence: 9999,
        data: b"hello mux world".to_vec(),
        flags: FLAG_SYN,
        timestamp: 1_000_000,
    };
    let encoded = original.encode();
    let decoded = StreamFrame::decode(&encoded).expect("decode");
    assert_eq!(decoded.stream_id, original.stream_id);
    assert_eq!(decoded.sequence, original.sequence);
    assert_eq!(decoded.data, original.data);
    assert_eq!(decoded.flags, original.flags);
    assert_eq!(decoded.timestamp, original.timestamp);
}

// ── T55: StreamFrame decode rejects truncated input ───────────────────
#[test]
fn t55_frame_decode_too_short() {
    let err = StreamFrame::decode(&[0u8; 10]).unwrap_err();
    assert!(matches!(err, MuxError::FrameTooLarge(10)));
}

// ── T56: StreamFrame encode/decode with empty payload ─────────────────
#[test]
fn t56_frame_encode_decode_empty_payload() {
    let frame = StreamFrame {
        stream_id: StreamId(1),
        sequence: 0,
        data: Vec::new(),
        flags: FLAG_FIN,
        timestamp: 0,
    };
    let encoded = frame.encode();
    let decoded = StreamFrame::decode(&encoded).expect("decode empty");
    assert!(decoded.data.is_empty());
    assert_eq!(decoded.flags, FLAG_FIN);
}

// ── T57: drain_recv_buffer returns buffered payloads ──────────────────
#[test]
fn t57_drain_recv_buffer() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");

    // Feed two DATA frames via receive_events.
    for seq in 0..2u64 {
        let payload = format!("payload-{seq}").into_bytes();
        let frame = StreamFrame {
            stream_id: id,
            sequence: seq,
            data: payload.clone(),
            flags: 0,
            timestamp: 0,
        };
        mux.receive_events(frame, 0).expect("receive_events");
    }

    let drained = mux.drain_recv_buffer(id).expect("drain");
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0], b"payload-0");
    assert_eq!(drained[1], b"payload-1");

    // Buffer is now empty.
    let empty = mux.drain_recv_buffer(id).expect("drain again");
    assert!(empty.is_empty());
}

// ── T58: drain_recv_buffer on unknown stream ───────────────────────────
#[test]
fn t58_drain_recv_buffer_missing() {
    let mut mux = default_mux();
    let err = mux.drain_recv_buffer(StreamId(99)).unwrap_err();
    assert_eq!(err, MuxError::StreamNotFound(99));
}

// ── T59: drain_send_queue drains all frames ───────────────────────────
#[test]
fn t59_drain_send_queue() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        max_frame_size: 4,
        default_window_size: 65536,
        ..MultiplexerConfig::default()
    });
    let a = mux
        .open_stream(StreamPriority::Low)
        .expect("test: open_stream should succeed");
    let b = mux
        .open_stream(StreamPriority::High)
        .expect("test: open_stream should succeed");
    mux.send(a, b"AAAA".to_vec(), 0)
        .expect("test: send should succeed");
    mux.send(b, b"BBBB".to_vec(), 0)
        .expect("test: send should succeed");
    // 2 SYN + 2 data = 4 frames total
    let frames = mux.drain_send_queue();
    assert_eq!(frames.len(), 4);
    // Queue is now empty.
    assert!(mux.drain_send_queue().is_empty());
}

// ── T60: drain_send_queue respects priority order ─────────────────────
#[test]
fn t60_drain_send_queue_priority_order() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        enable_priority: true,
        ..MultiplexerConfig::default()
    });
    let lo = mux
        .open_stream(StreamPriority::Background)
        .expect("test: open_stream should succeed");
    let hi = mux
        .open_stream(StreamPriority::Critical)
        .expect("test: open_stream should succeed");
    // Drain SYN frames (they have different priorities)
    mux.drain_send_queue();

    mux.send(lo, b"low".to_vec(), 0)
        .expect("test: send should succeed");
    mux.send(hi, b"high".to_vec(), 0)
        .expect("test: send should succeed");

    let frames = mux.drain_send_queue();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].stream_id, hi); // Critical first
    assert_eq!(frames[1].stream_id, lo);
}

// ── T61: expire_idle closes inactive streams ───────────────────────────
#[test]
fn t61_expire_idle_closes_streams() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        idle_timeout_us: 1000,
        ..MultiplexerConfig::default()
    });
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    // last_activity = 0, idle_timeout = 1000 → idle if current_ts > 1000
    let events = mux.expire_idle(2000);
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], MuxEvent::IdleStreamExpired(_)));
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .state,
        StreamState::Closed
    );
}

// ── T62: expire_idle keeps recently active streams ────────────────────
#[test]
fn t62_expire_idle_keeps_active() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        idle_timeout_us: 10_000,
        ..MultiplexerConfig::default()
    });
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    // Simulate activity at ts=5000
    mux.send(id, b"ping".to_vec(), 5000)
        .expect("test: send should succeed");
    // current_ts = 6000; last_activity(5000) + timeout(10000) = 15000 > 6000 → no expiry
    let events = mux.expire_idle(6000);
    assert!(events.is_empty());
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .state,
        StreamState::Open
    );
}

// ── T63: active_streams returns only Open streams ─────────────────────
#[test]
fn t63_active_streams() {
    let mut mux = default_mux();
    let a = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let b = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.reset_stream(a)
        .expect("test: reset_stream should succeed");

    let active = mux.active_streams();
    assert_eq!(active.len(), 1);
    assert!(active.contains(&b));
}

// ── T64: stream_info returns correct snapshot ──────────────────────────
#[test]
fn t64_stream_info() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::High)
        .expect("test: open_stream should succeed");
    mux.send(id, b"hello".to_vec(), 0)
        .expect("test: send should succeed");
    let info: StreamInfo = mux.stream_info(id).expect("info");
    assert_eq!(info.id, id);
    assert_eq!(info.state, StreamState::Open);
    assert_eq!(info.bytes_sent, 5);
}

// ── T65: stream_info errors on unknown stream ──────────────────────────
#[test]
fn t65_stream_info_missing() {
    let mux = default_mux();
    let err = mux.stream_info(StreamId(55)).unwrap_err();
    assert_eq!(err, MuxError::StreamNotFound(55));
}

// ── T66: FrameFlags typed bitfield operations ──────────────────────────
#[test]
fn t66_frame_flags_bitfield() {
    let flags = FrameFlags::new(FrameFlags::SYN | FrameFlags::DATA);
    assert!(flags.is_syn());
    assert!(flags.is_data());
    assert!(!flags.is_fin());
    assert!(!flags.is_rst());
    assert!(!flags.is_ack());

    let all = FrameFlags::new(
        FrameFlags::SYN | FrameFlags::FIN | FrameFlags::RST | FrameFlags::ACK | FrameFlags::DATA,
    );
    assert!(all.is_syn());
    assert!(all.is_fin());
    assert!(all.is_rst());
    assert!(all.is_ack());
    assert!(all.is_data());
}

// ── T67: FrameFlags Display ────────────────────────────────────────────
#[test]
fn t67_frame_flags_display() {
    let none = FrameFlags::new(0);
    assert_eq!(format!("{none}"), "NONE");

    let syn = FrameFlags::new(FrameFlags::SYN);
    let s = format!("{syn}");
    assert!(s.contains("SYN"));

    let multi = FrameFlags::new(FrameFlags::FIN | FrameFlags::ACK);
    let s2 = format!("{multi}");
    assert!(s2.contains("FIN"));
    assert!(s2.contains("ACK"));
}

// ── T68: receive_events SYN opens a stream ─────────────────────────────
#[test]
fn t68_receive_events_syn_opens_stream() {
    let mut mux = default_mux();
    let new_id = StreamId(100);
    let frame = StreamFrame {
        stream_id: new_id,
        sequence: 0,
        data: Vec::new(),
        flags: FLAG_SYN,
        timestamp: 0,
    };
    let events = mux.receive_events(frame, 0).expect("events");
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], MuxEvent::StreamOpened(id) if id == new_id));
    assert!(mux.stream_state(new_id).is_some());
}

// ── T69: receive_events FIN closes stream ─────────────────────────────
#[test]
fn t69_receive_events_fin_closes_stream() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = StreamFrame {
        stream_id: id,
        sequence: 0,
        data: Vec::new(),
        flags: FLAG_FIN,
        timestamp: 0,
    };
    let events = mux.receive_events(frame, 0).expect("events");
    assert!(events
        .iter()
        .any(|e| matches!(e, MuxEvent::StreamClosed(_))));
}

// ── T70: receive_events RST resets stream ─────────────────────────────
#[test]
fn t70_receive_events_rst_resets_stream() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = StreamFrame {
        stream_id: id,
        sequence: 42, // ignored for RST
        data: Vec::new(),
        flags: FLAG_RST,
        timestamp: 0,
    };
    let events = mux.receive_events(frame, 0).expect("events");
    assert!(events.iter().any(|e| matches!(e, MuxEvent::StreamReset(_))));
    assert_eq!(
        mux.stream_state(id)
            .expect("test: stream should exist")
            .state,
        StreamState::Reset
    );
}

// ── T71: xorshift64 produces non-zero sequence ────────────────────────
#[test]
fn t71_xorshift64_basic() {
    let mut state = 1u64;
    let v1 = xorshift64(&mut state);
    let v2 = xorshift64(&mut state);
    let v3 = xorshift64(&mut state);
    // Values must be non-zero and all distinct for a good seed.
    assert_ne!(v1, 0);
    assert_ne!(v2, 0);
    assert_ne!(v3, 0);
    assert_ne!(v1, v2);
    assert_ne!(v2, v3);
}

// ── T72: xorshift64 is deterministic ─────────────────────────────────
#[test]
fn t72_xorshift64_deterministic() {
    let mut s1 = 0xDEAD_BEEF_u64;
    let mut s2 = 0xDEAD_BEEF_u64;
    for _ in 0..100 {
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }
}

// ── T73: priority_from_u8 mapping ─────────────────────────────────────
#[test]
fn t73_priority_from_u8() {
    assert_eq!(priority_from_u8(0), StreamPriority::Background);
    assert_eq!(priority_from_u8(1), StreamPriority::Low);
    assert_eq!(priority_from_u8(63), StreamPriority::Low);
    assert_eq!(priority_from_u8(64), StreamPriority::Normal);
    assert_eq!(priority_from_u8(127), StreamPriority::Normal);
    assert_eq!(priority_from_u8(128), StreamPriority::High);
    assert_eq!(priority_from_u8(191), StreamPriority::High);
    assert_eq!(priority_from_u8(192), StreamPriority::Critical);
    assert_eq!(priority_from_u8(255), StreamPriority::Critical);
}

// ── T74: open_stream_with_priority preserves raw priority ─────────────
#[test]
fn t74_open_stream_with_priority_raw() {
    let mut mux = default_mux();
    let id = mux
        .open_stream_with_priority(200)
        .expect("test: open_stream_with_priority should succeed");
    let stream = mux.stream_state(id).expect("test: stream should exist");
    assert_eq!(stream.priority_raw, 200);
    assert_eq!(stream.priority, StreamPriority::Critical);
}

// ── T75: multiplexer_stats tracks total_streams_opened ────────────────
#[test]
fn t75_multiplexer_stats_total_opened() {
    let mut mux = default_mux();
    mux.open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    mux.open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let stats: MultiplexerStats = mux.multiplexer_stats();
    assert_eq!(stats.total_streams_opened, 2);
    assert_eq!(stats.active_streams, 2);
}

// ── T76: multiplexer_stats tracks bytes_received ───────────────────────
#[test]
fn t76_multiplexer_stats_bytes_received() {
    let mut mux = default_mux();
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");
    let frame = StreamFrame {
        stream_id: id,
        sequence: 0,
        data: b"12345".to_vec(),
        flags: 0,
        timestamp: 0,
    };
    mux.receive(frame, 0).expect("test: receive should succeed");
    let stats = mux.multiplexer_stats();
    assert_eq!(stats.total_bytes_received, 5);
}

// ── T77: large fragmentation test with xorshift64 ─────────────────────
#[test]
fn t77_large_fragmentation_xorshift() {
    let mut mux = StreamMultiplexer::new(MultiplexerConfig {
        max_frame_size: 64,
        default_window_size: 65536,
        ..MultiplexerConfig::default()
    });
    let id = mux
        .open_stream(StreamPriority::Normal)
        .expect("test: open_stream should succeed");

    // Generate 1024 pseudo-random bytes.
    let mut rng_state = 0xCAFE_BABE_u64;
    let payload: Vec<u8> = (0..1024)
        .map(|_| (xorshift64(&mut rng_state) & 0xFF) as u8)
        .collect();

    let n = mux
        .send(id, payload.clone(), 0)
        .expect("test: send should succeed");
    assert_eq!(n, 1024);

    // Collect all data frames and verify reconstruction.
    let frames = mux.drain_send_queue();
    let reconstructed: Vec<u8> = frames
        .iter()
        .filter(|f| !f.is_control())
        .flat_map(|f| f.data.iter().copied())
        .collect();
    assert_eq!(reconstructed, payload);
}

// ── T78: encode/decode preserves large payload ─────────────────────────
#[test]
fn t78_encode_decode_large_payload() {
    let mut rng = 0xABCD_1234_u64;
    let payload: Vec<u8> = (0..4096)
        .map(|_| (xorshift64(&mut rng) & 0xFF) as u8)
        .collect();
    let frame = StreamFrame {
        stream_id: StreamId(7),
        sequence: 123456,
        data: payload.clone(),
        flags: 0,
        timestamp: 999_999,
    };
    let encoded = frame.encode();
    let decoded = StreamFrame::decode(&encoded).expect("test: decode should succeed");
    assert_eq!(decoded.data, payload);
    assert_eq!(decoded.sequence, 123456);
    assert_eq!(decoded.timestamp, 999_999);
}

// ── T79: StreamId Display ─────────────────────────────────────────────
#[test]
fn t79_stream_id_display() {
    let id = StreamId(42);
    assert_eq!(format!("{id}"), "stream:42");
}

// ── T80: StreamState Opening variant exists ────────────────────────────
#[test]
fn t80_stream_state_opening_variant() {
    let state = StreamState::Opening;
    let debug = format!("{state:?}");
    assert!(debug.contains("Opening"));
}
