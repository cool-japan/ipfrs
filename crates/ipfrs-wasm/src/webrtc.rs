//! WebRTC transport for browser-to-browser peer connectivity.
//!
//! This module provides a WebRTC data-channel-based block exchange mechanism
//! for IPFRS in the browser.  After initial out-of-band signalling (offer/answer
//! SDP + ICE candidates), peers can exchange content-addressed blocks directly
//! without a server relay.
//!
//! # Architecture
//!
//! ```text
//!   Caller                                 Answerer
//!   ──────                                 ────────
//!   IpfrsPeer::new(id)                     IpfrsPeerAnswerer::from_offer(offer_sdp)
//!   create_offer() ─── sdp offer ────────► (set as remote desc internally)
//!                                          create_answer() ─► answer_sdp
//!   set_answer(answer_sdp) ◄─ sdp answer ─
//!   add_ice_candidate(…)   ◄─► ICE  ────► add_ice_candidate(…)
//!   [DataChannel open]
//!   send_block(cid, data) ──────────────► [DataChannel message received by JS]
//! ```
//!
//! # Signal types
//!
//! [`WebRtcSignal`] and [`IceCandidate`] are pure-Rust structs that are
//! serialisable to / from JSON via `serde_json` and can be transported over
//! *any* signalling channel (WebSocket, HTTP, etc.) chosen by the application.
//!
//! # WASM gating
//!
//! `IpfrsPeer` and `IpfrsPeerAnswerer` wrap `web_sys` types that are only
//! available inside a browser (WASM target).  All code that references `web_sys`
//! is therefore placed inside `#[cfg(target_arch = "wasm32")]` blocks so that
//! the signalling types remain usable in native unit tests.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Signalling data types  (target-independent)
// ---------------------------------------------------------------------------

/// An ICE candidate produced by the local peer that must be forwarded to the
/// remote peer over the signalling channel.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct IceCandidate {
    /// The SDP candidate string (e.g. `"candidate:0 1 UDP 2122252543 192.168.1.1 …"`).
    pub candidate: String,
    /// The media stream identification tag (`mid`) identifying the media
    /// description this candidate is associated with, if any.
    pub sdp_mid: Option<String>,
    /// Zero-based index of the media description in the SDP, if any.
    pub sdp_m_line_index: Option<u16>,
}

/// A signalling message exchanged during WebRTC connection establishment.
///
/// Applications must forward these messages to the remote peer over a
/// signalling channel of their choice (WebSocket, HTTP POST, etc.).
///
/// The `type` field in the JSON representation is the enum variant name:
/// `"Offer"`, `"Answer"`, or `"IceCandidate"`.
///
/// ```json
/// {"type":"Offer","sdp":"v=0\r\no=- …"}
/// {"type":"Answer","sdp":"v=0\r\no=- …"}
/// {"type":"IceCandidate","candidate":"candidate:0 …","sdp_mid":"0","sdp_m_line_index":0}
/// ```
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type")]
pub enum WebRtcSignal {
    /// SDP offer created by the caller peer.
    Offer { sdp: String },
    /// SDP answer created by the answerer peer.
    Answer { sdp: String },
    /// ICE candidate from either peer.
    IceCandidate(IceCandidate),
}

// ---------------------------------------------------------------------------
// WASM-only WebRTC peer implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
mod wasm_peer {
    use super::{IceCandidate, WebRtcSignal};
    use js_sys::{Object, Reflect};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{
        RtcConfiguration, RtcDataChannel, RtcDataChannelInit, RtcDataChannelState, RtcIceCandidate,
        RtcIceCandidateInit, RtcPeerConnection, RtcSdpType, RtcSessionDescriptionInit,
    };

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Create a default [`RtcPeerConnection`] with no ICE servers configured.
    ///
    /// ICE servers (STUN/TURN) should be configured by the application via a
    /// more elaborate configuration if NAT traversal is required.
    fn create_peer_connection() -> Result<RtcPeerConnection, JsValue> {
        let config = RtcConfiguration::new();
        RtcPeerConnection::new_with_configuration(&config)
            .map_err(|e| JsValue::from_str(&format!("RtcPeerConnection::new failed: {e:?}")))
    }

    /// Build an `RtcSessionDescriptionInit` from a raw SDP string and type.
    fn make_sdp_init(sdp_type: RtcSdpType, sdp: &str) -> RtcSessionDescriptionInit {
        let desc = RtcSessionDescriptionInit::new(sdp_type);
        desc.set_sdp(sdp);
        desc
    }

    /// Deserialise an [`IceCandidate`] from a JSON string and add it to `pc`.
    async fn apply_ice_candidate(
        pc: &RtcPeerConnection,
        candidate_json: &str,
    ) -> Result<(), JsValue> {
        let ice: IceCandidate = serde_json::from_str(candidate_json)
            .map_err(|e| JsValue::from_str(&format!("IceCandidate JSON parse error: {e}")))?;

        let mut init = RtcIceCandidateInit::new(&ice.candidate);
        if let Some(mid) = &ice.sdp_mid {
            init.sdp_mid(Some(mid.as_str()));
        }
        if let Some(idx) = ice.sdp_m_line_index {
            init.sdp_m_line_index(Some(idx));
        }

        let rtc_ice = RtcIceCandidate::new(&init)
            .map_err(|e| JsValue::from_str(&format!("RtcIceCandidate::new failed: {e:?}")))?;

        JsFuture::from(
            pc.add_ice_candidate_with_opt_rtc_ice_candidate(Some(&rtc_ice))
                .map_err(|e| JsValue::from_str(&format!("addIceCandidate failed: {e:?}")))?,
        )
        .await
        .map(|_| ())
    }

    // -----------------------------------------------------------------------
    // IpfrsPeer – caller side (creates offer, owns the DataChannel)
    // -----------------------------------------------------------------------

    /// Browser-to-browser peer connection handle — *caller* side.
    ///
    /// The caller creates an SDP offer, receives an SDP answer from the remote
    /// peer through the signalling channel, and exchanges ICE candidates until
    /// the data channel is open.
    ///
    /// # JavaScript example
    ///
    /// ```javascript
    /// const peer = new IpfrsPeer("peer-a");
    /// const offerSdp = await peer.create_offer();
    ///
    /// // … send offerSdp to the answerer via your signalling channel …
    ///
    /// const answerSdp = /* received from answerer */;
    /// await peer.set_answer(answerSdp);
    ///
    /// // … exchange ICE candidates …
    ///
    /// // Once is_connected() returns true:
    /// peer.send_block("bafk…", new Uint8Array([1, 2, 3]));
    /// ```
    #[wasm_bindgen]
    pub struct IpfrsPeer {
        peer_id: String,
        /// The underlying RTCPeerConnection.
        #[wasm_bindgen(skip)]
        inner: RtcPeerConnection,
        /// The DataChannel created by the caller (negotiated in-band).
        #[wasm_bindgen(skip)]
        data_channel: Option<RtcDataChannel>,
    }

    #[wasm_bindgen]
    impl IpfrsPeer {
        /// Create a new caller-side peer connection.
        ///
        /// This allocates an `RTCPeerConnection` and a negotiation data channel
        /// labelled `"ipfrs-blocks"`.  Call [`IpfrsPeer::create_offer`] next to
        /// generate the SDP offer.
        ///
        /// # Parameters
        /// - `peer_id` — an application-level identifier for this peer; returned
        ///   by [`IpfrsPeer::peer_id`] for routing purposes.
        #[wasm_bindgen(constructor)]
        pub fn new(peer_id: &str) -> Result<IpfrsPeer, JsValue> {
            let pc = create_peer_connection()?;

            // Create the data channel on the caller side; the answerer receives
            // it via the `ondatachannel` event.
            let mut dc_init = RtcDataChannelInit::new();
            dc_init.ordered(true);

            let dc = pc
                .create_data_channel_with_data_channel_dict("ipfrs-blocks", &dc_init)
                .map_err(|e| JsValue::from_str(&format!("createDataChannel failed: {e:?}")))?;

            Ok(IpfrsPeer {
                peer_id: peer_id.to_string(),
                inner: pc,
                data_channel: Some(dc),
            })
        }

        /// Generate an SDP offer and set it as the local description.
        ///
        /// Returns the offer SDP string.  Forward this to the answerer peer via
        /// your signalling channel.
        pub async fn create_offer(&self) -> Result<String, JsValue> {
            let offer_promise = self
                .inner
                .create_offer()
                .map_err(|e| JsValue::from_str(&format!("createOffer() failed: {e:?}")))?;

            let offer_value = JsFuture::from(offer_promise).await?;

            // Extract the SDP string from the RTCSessionDescriptionInit object.
            let sdp = Reflect::get(&offer_value, &JsValue::from_str("sdp"))
                .map_err(|_| JsValue::from_str("offer missing 'sdp' field"))?
                .as_string()
                .ok_or_else(|| JsValue::from_str("offer 'sdp' is not a string"))?;

            // Set local description.
            let local_desc = make_sdp_init(RtcSdpType::Offer, &sdp);
            JsFuture::from(
                self.inner.set_local_description(&local_desc).map_err(|e| {
                    JsValue::from_str(&format!("setLocalDescription failed: {e:?}"))
                })?,
            )
            .await?;

            Ok(sdp)
        }

        /// Apply the SDP answer received from the remote (answerer) peer.
        ///
        /// Call this after receiving the answer SDP through your signalling
        /// channel.  ICE candidate exchange may begin before or after this call.
        pub async fn set_answer(&self, answer_sdp: &str) -> Result<(), JsValue> {
            let remote_desc = make_sdp_init(RtcSdpType::Answer, answer_sdp);
            JsFuture::from(
                self.inner
                    .set_remote_description(&remote_desc)
                    .map_err(|e| {
                        JsValue::from_str(&format!("setRemoteDescription failed: {e:?}"))
                    })?,
            )
            .await
            .map(|_| ())
        }

        /// Add an ICE candidate received from the remote peer.
        ///
        /// `candidate_json` must be a JSON-serialised [`IceCandidate`].
        pub fn add_ice_candidate(&self, candidate_json: &str) -> Result<(), JsValue> {
            // Parse the candidate first to validate it before spawning the async.
            let ice: IceCandidate = serde_json::from_str(candidate_json)
                .map_err(|e| JsValue::from_str(&format!("IceCandidate JSON parse error: {e}")))?;

            let mut init = RtcIceCandidateInit::new(&ice.candidate);
            if let Some(mid) = &ice.sdp_mid {
                init.sdp_mid(Some(mid.as_str()));
            }
            if let Some(idx) = ice.sdp_m_line_index {
                init.sdp_m_line_index(Some(idx));
            }

            let rtc_ice = RtcIceCandidate::new(&init)
                .map_err(|e| JsValue::from_str(&format!("RtcIceCandidate::new: {e:?}")))?;

            // Fire-and-forget via wasm_bindgen_futures; errors are surfaced in
            // the browser console.
            let pc = self.inner.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Ok(promise) = pc.add_ice_candidate_with_opt_rtc_ice_candidate(Some(&rtc_ice))
                {
                    let _ = JsFuture::from(promise).await;
                }
            });

            Ok(())
        }

        /// Send a content-addressed block to the remote peer.
        ///
        /// The wire format is:
        /// ```text
        /// [4-byte big-endian CID length][CID UTF-8 bytes][block data bytes]
        /// ```
        ///
        /// The DataChannel must be open (`is_connected()` returns `true`) before
        /// calling this method.
        pub fn send_block(&self, cid: &str, data: &[u8]) -> Result<(), JsValue> {
            let dc = self
                .data_channel
                .as_ref()
                .ok_or_else(|| JsValue::from_str("DataChannel not initialised"))?;

            if dc.ready_state() != RtcDataChannelState::Open {
                return Err(JsValue::from_str(
                    "DataChannel is not open; wait for is_connected() == true",
                ));
            }

            // Encode: [u32 BE cid_len][cid bytes][data bytes]
            let cid_bytes = cid.as_bytes();
            let cid_len = cid_bytes.len() as u32;
            let mut buf = Vec::with_capacity(4 + cid_bytes.len() + data.len());
            buf.extend_from_slice(&cid_len.to_be_bytes());
            buf.extend_from_slice(cid_bytes);
            buf.extend_from_slice(data);

            dc.send_with_u8_array(&buf)
                .map_err(|e| JsValue::from_str(&format!("DataChannel.send failed: {e:?}")))
        }

        /// Return the application-level peer identifier supplied to [`IpfrsPeer::new`].
        pub fn peer_id(&self) -> String {
            self.peer_id.clone()
        }

        /// Return `true` when the data channel is in the `"open"` state and
        /// ready to transmit blocks.
        pub fn is_connected(&self) -> bool {
            self.data_channel
                .as_ref()
                .map(|dc| dc.ready_state() == RtcDataChannelState::Open)
                .unwrap_or(false)
        }
    }

    // -----------------------------------------------------------------------
    // IpfrsPeerAnswerer – answerer side (receives offer, creates answer)
    // -----------------------------------------------------------------------

    /// Browser-to-browser peer connection handle — *answerer* side.
    ///
    /// The answerer receives an SDP offer from the caller through a signalling
    /// channel, creates an answer SDP, and exchanges ICE candidates.
    ///
    /// # JavaScript example
    ///
    /// ```javascript
    /// const offerSdp = /* received from caller via your signalling channel */;
    /// const answerer = await IpfrsPeerAnswerer.from_offer(offerSdp);
    /// const answerSdp = await answerer.create_answer();
    ///
    /// // … send answerSdp back to the caller via your signalling channel …
    ///
    /// // … exchange ICE candidates …
    ///
    /// // When is_connected() returns true, the data channel is ready.
    /// ```
    #[wasm_bindgen]
    pub struct IpfrsPeerAnswerer {
        /// The underlying RTCPeerConnection.
        #[wasm_bindgen(skip)]
        inner: RtcPeerConnection,
    }

    #[wasm_bindgen]
    impl IpfrsPeerAnswerer {
        /// Create an answerer peer connection from the caller's SDP offer.
        ///
        /// Sets the offer as the remote description on the newly created
        /// `RTCPeerConnection`.
        pub async fn from_offer(offer_sdp: &str) -> Result<IpfrsPeerAnswerer, JsValue> {
            let pc = create_peer_connection()?;

            let remote_desc = make_sdp_init(RtcSdpType::Offer, offer_sdp);
            JsFuture::from(pc.set_remote_description(&remote_desc).map_err(|e| {
                JsValue::from_str(&format!("setRemoteDescription(offer) failed: {e:?}"))
            })?)
            .await?;

            Ok(IpfrsPeerAnswerer { inner: pc })
        }

        /// Generate an SDP answer and set it as the local description.
        ///
        /// Returns the answer SDP string.  Forward this to the caller via your
        /// signalling channel.
        pub async fn create_answer(&self) -> Result<String, JsValue> {
            let answer_promise = self
                .inner
                .create_answer()
                .map_err(|e| JsValue::from_str(&format!("createAnswer() failed: {e:?}")))?;

            let answer_value = JsFuture::from(answer_promise).await?;

            let sdp = Reflect::get(&answer_value, &JsValue::from_str("sdp"))
                .map_err(|_| JsValue::from_str("answer missing 'sdp' field"))?
                .as_string()
                .ok_or_else(|| JsValue::from_str("answer 'sdp' is not a string"))?;

            let local_desc = make_sdp_init(RtcSdpType::Answer, &sdp);
            JsFuture::from(
                self.inner.set_local_description(&local_desc).map_err(|e| {
                    JsValue::from_str(&format!("setLocalDescription failed: {e:?}"))
                })?,
            )
            .await?;

            Ok(sdp)
        }

        /// Add an ICE candidate received from the remote (caller) peer.
        ///
        /// `candidate_json` must be a JSON-serialised [`IceCandidate`].
        pub async fn add_ice_candidate(&self, candidate_json: &str) -> Result<(), JsValue> {
            apply_ice_candidate(&self.inner, candidate_json).await
        }

        /// Return `true` when the ICE connection has reached a usable state
        /// (`"connected"` or `"completed"`).
        pub fn is_connected(&self) -> bool {
            use web_sys::RtcIceConnectionState;
            let state = self.inner.ice_connection_state();
            state == RtcIceConnectionState::Connected || state == RtcIceConnectionState::Completed
        }
    }

    // Re-export so they're accessible from the crate root under `webrtc::`.
    pub use IpfrsPeer;
    pub use IpfrsPeerAnswerer;
}

// ---------------------------------------------------------------------------
// Re-export WASM types at module level (wasm32 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
pub use wasm_peer::{IpfrsPeer, IpfrsPeerAnswerer};

// ---------------------------------------------------------------------------
// Tests – run on native (no web_sys needed)
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::{IceCandidate, WebRtcSignal};

    // -----------------------------------------------------------------------
    // WebRtcSignal::Offer roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_webrtc_signal_offer_roundtrip() {
        let original = WebRtcSignal::Offer {
            sdp: "v=0\r\no=- 1234 2 IN IP4 127.0.0.1\r\n".to_string(),
        };

        let json = serde_json::to_string(&original).expect("serialise Offer");
        assert!(
            json.contains("\"type\":\"Offer\""),
            "type tag missing: {json}"
        );
        assert!(json.contains("\"sdp\""), "sdp field missing: {json}");

        let roundtripped: WebRtcSignal = serde_json::from_str(&json).expect("deserialise Offer");
        assert_eq!(original, roundtripped);
    }

    // -----------------------------------------------------------------------
    // WebRtcSignal::Answer roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_webrtc_signal_answer_roundtrip() {
        let original = WebRtcSignal::Answer {
            sdp: "v=0\r\no=- 5678 2 IN IP4 127.0.0.1\r\n".to_string(),
        };

        let json = serde_json::to_string(&original).expect("serialise Answer");
        assert!(
            json.contains("\"type\":\"Answer\""),
            "type tag missing: {json}"
        );

        let roundtripped: WebRtcSignal = serde_json::from_str(&json).expect("deserialise Answer");
        assert_eq!(original, roundtripped);
    }

    // -----------------------------------------------------------------------
    // IceCandidate roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_webrtc_signal_ice_candidate() {
        let ice = IceCandidate {
            candidate: "candidate:0 1 UDP 2122252543 192.168.1.42 54321 typ host".to_string(),
            sdp_mid: Some("0".to_string()),
            sdp_m_line_index: Some(0),
        };
        let signal = WebRtcSignal::IceCandidate(ice.clone());

        let json = serde_json::to_string(&signal).expect("serialise IceCandidate");
        assert!(
            json.contains("\"type\":\"IceCandidate\""),
            "type tag missing: {json}"
        );
        assert!(
            json.contains("\"candidate\""),
            "candidate field missing: {json}"
        );

        let roundtripped: WebRtcSignal =
            serde_json::from_str(&json).expect("deserialise IceCandidate");

        match roundtripped {
            WebRtcSignal::IceCandidate(rt_ice) => {
                assert_eq!(rt_ice.candidate, ice.candidate);
                assert_eq!(rt_ice.sdp_mid, ice.sdp_mid);
                assert_eq!(rt_ice.sdp_m_line_index, ice.sdp_m_line_index);
            }
            other => panic!("expected IceCandidate, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // IceCandidate with None fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_ice_candidate_none_fields_roundtrip() {
        let ice = IceCandidate {
            candidate: "candidate:1 1 TCP 1518280447 ::1 9 typ host tcptype active".to_string(),
            sdp_mid: None,
            sdp_m_line_index: None,
        };
        let json = serde_json::to_string(&ice).expect("serialise IceCandidate None");
        let rt: IceCandidate = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(rt, ice);
        assert!(rt.sdp_mid.is_none());
        assert!(rt.sdp_m_line_index.is_none());
    }

    // -----------------------------------------------------------------------
    // type tag format verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_signal_type_tag() {
        let offer = WebRtcSignal::Offer {
            sdp: "dummy".to_string(),
        };
        let json = serde_json::to_string(&offer).expect("serialise");
        // Verify the exact key/value at the start
        assert!(
            json.starts_with(r#"{"type":"Offer""#),
            "JSON must start with {{\"type\":\"Offer\": got {json}"
        );

        let answer = WebRtcSignal::Answer {
            sdp: "dummy".to_string(),
        };
        let json_a = serde_json::to_string(&answer).expect("serialise");
        assert!(
            json_a.starts_with(r#"{"type":"Answer""#),
            "JSON must start with {{\"type\":\"Answer\": got {json_a}"
        );

        let ice = WebRtcSignal::IceCandidate(IceCandidate {
            candidate: "c".to_string(),
            sdp_mid: None,
            sdp_m_line_index: None,
        });
        let json_i = serde_json::to_string(&ice).expect("serialise");
        assert!(
            json_i.starts_with(r#"{"type":"IceCandidate""#),
            "JSON must start with {{\"type\":\"IceCandidate\": got {json_i}"
        );
    }

    // -----------------------------------------------------------------------
    // Deserialise from known JSON strings (forward-compat check)
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialise_known_offer_json() {
        // In JSON, "\r\n" is an escape sequence for actual CR+LF characters.
        let json = "{\"type\":\"Offer\",\"sdp\":\"v=0\\r\\n\"}";
        let signal: WebRtcSignal = serde_json::from_str(json).expect("parse known Offer JSON");
        match signal {
            WebRtcSignal::Offer { sdp } => {
                assert!(
                    sdp.starts_with("v=0"),
                    "SDP must start with v=0, got: {sdp:?}"
                );
                // The JSON "\r\n" decodes to actual CR+LF bytes.
                assert!(sdp.contains('\r'), "SDP must contain CR after JSON decode");
                assert!(sdp.contains('\n'), "SDP must contain LF after JSON decode");
            }
            other => panic!("expected Offer, got {other:?}"),
        }
    }

    #[test]
    fn test_deserialise_known_ice_json() {
        let json = r#"{"type":"IceCandidate","candidate":"candidate:0 1 UDP 123 10.0.0.1 5000 typ host","sdp_mid":"audio","sdp_m_line_index":0}"#;
        let signal: WebRtcSignal = serde_json::from_str(json).expect("parse known ICE JSON");
        match signal {
            WebRtcSignal::IceCandidate(ice) => {
                assert!(ice.candidate.starts_with("candidate:0"));
                assert_eq!(ice.sdp_mid.as_deref(), Some("audio"));
                assert_eq!(ice.sdp_m_line_index, Some(0));
            }
            other => panic!("expected IceCandidate, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Clone / Debug derives
    // -----------------------------------------------------------------------

    #[test]
    fn test_signal_clone_and_debug() {
        let sig = WebRtcSignal::Offer {
            sdp: "test".to_string(),
        };
        let cloned = sig.clone();
        assert_eq!(sig, cloned);
        // Debug must not panic
        let _ = format!("{sig:?}");
    }

    #[test]
    fn test_ice_candidate_clone_and_debug() {
        let ice = IceCandidate {
            candidate: "cand".to_string(),
            sdp_mid: Some("m".to_string()),
            sdp_m_line_index: Some(1),
        };
        let cloned = ice.clone();
        assert_eq!(ice, cloned);
        let _ = format!("{ice:?}");
    }
}
