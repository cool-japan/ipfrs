//! Fuzz test for IPLD parsing
//!
//! Tests robustness of IPLD deserialization against arbitrary data

#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::Ipld;

fuzz_target!(|data: &[u8]| {
    // Try to deserialize as DAG-CBOR
    let _ = Ipld::from_dag_cbor(data);

    // Try to deserialize as DAG-JSON (requires valid UTF-8)
    if let Ok(json_str) = std::str::from_utf8(data) {
        let _ = Ipld::from_dag_json(json_str);
    }

    // If DAG-CBOR parsing succeeds, test round-trip
    if let Ok(ipld) = Ipld::from_dag_cbor(data) {
        // Serialize back to DAG-CBOR
        if let Ok(cbor) = ipld.to_dag_cbor() {
            // Parse again
            if let Ok(ipld2) = Ipld::from_dag_cbor(&cbor) {
                // Should be identical
                assert_eq!(ipld, ipld2);
            }
        }
    }

    // If DAG-JSON parsing succeeds, test round-trip
    if let Ok(json_str) = std::str::from_utf8(data) {
        if let Ok(ipld) = Ipld::from_dag_json(json_str) {
            // Serialize back to DAG-JSON
            if let Ok(json) = ipld.to_dag_json() {
                // Parse again
                if let Ok(ipld2) = Ipld::from_dag_json(&json) {
                    // Should be identical
                    assert_eq!(ipld, ipld2);
                }
            }
        }
    }
});
