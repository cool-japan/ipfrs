#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::{global_codec_registry, Ipld};
use std::collections::BTreeMap;

fuzz_target!(|data: &[u8]| {
    let registry = global_codec_registry();

    // Test DAG-CBOR codec
    if data.len() > 0 && data.len() < 1000 {
        // Try decoding arbitrary data as DAG-CBOR
        let _ = registry.decode(0x71, data);

        // Create IPLD from data and test roundtrip
        let ipld = Ipld::Bytes(data.to_vec());
        if let Ok(encoded) = registry.encode(0x71, &ipld) {
            if let Ok(decoded) = registry.decode(0x71, &encoded) {
                assert_eq!(ipld, decoded);
            }
        }
    }

    // Test DAG-JSON codec
    if let Ok(s) = std::str::from_utf8(data) {
        // Try decoding arbitrary UTF-8 as JSON
        let _ = registry.decode(0x0129, s.as_bytes());
    }

    // Test with complex IPLD structures
    if data.len() >= 4 {
        let mut map = BTreeMap::new();
        map.insert("key".to_string(), Ipld::Integer(i128::from_le_bytes([
            data[0], data[1], data[2], data[3],
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ])));
        let ipld = Ipld::Map(map);

        // Test all codecs
        for codec in [0x55, 0x71, 0x0129] {
            if let Ok(encoded) = registry.encode(codec, &ipld) {
                let _ = registry.decode(codec, &encoded);
            }
        }
    }
});
