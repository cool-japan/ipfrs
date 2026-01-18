#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::{Ipld, JoseBuilder, JoseSignature};

fuzz_target!(|data: &[u8]| {
    // Fuzz DAG-JOSE encoding/decoding

    // Only proceed if we have enough data for a secret (32 bytes minimum)
    if data.len() < 32 {
        return;
    }

    // Split data into secret and payload
    let (secret, payload_data) = data.split_at(32);

    // Try to parse payload_data as IPLD-like data
    let ipld = match payload_data.first() {
        Some(&0) => Ipld::Null,
        Some(&1) if payload_data.len() > 1 => Ipld::Bool(payload_data[1] != 0),
        Some(&2) if payload_data.len() >= 9 => {
            let value = i64::from_le_bytes([
                payload_data[1],
                payload_data[2],
                payload_data[3],
                payload_data[4],
                payload_data[5],
                payload_data[6],
                payload_data[7],
                payload_data[8],
            ]);
            Ipld::Integer(value as i128)
        }
        Some(&3) if payload_data.len() >= 9 => {
            let value = f64::from_le_bytes([
                payload_data[1],
                payload_data[2],
                payload_data[3],
                payload_data[4],
                payload_data[5],
                payload_data[6],
                payload_data[7],
                payload_data[8],
            ]);
            if value.is_finite() {
                Ipld::Float(value)
            } else {
                Ipld::Null
            }
        }
        Some(&4) if payload_data.len() > 1 => {
            let string_data = &payload_data[1..];
            if let Ok(s) = std::str::from_utf8(string_data) {
                Ipld::String(s.to_string())
            } else {
                Ipld::Null
            }
        }
        Some(&5) if payload_data.len() > 1 => Ipld::Bytes(payload_data[1..].to_vec()),
        _ => Ipld::Null,
    };

    // Try to sign the IPLD data
    if let Ok(jose) = JoseBuilder::new().with_payload(ipld.clone()).sign_hs256(secret) {
        // Verify the signature
        let _ = jose.verify_hs256(secret);

        // Try to encode to DAG-JOSE
        if let Ok(dag_jose) = jose.to_dag_jose() {
            // Try to decode back
            let _ = JoseSignature::from_dag_jose(&dag_jose);
        }
    }

    // Also fuzz the DAG-JOSE parsing directly
    let _ = JoseSignature::from_dag_jose(payload_data);
});
