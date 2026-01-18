#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs::Ipld;

fuzz_target!(|data: &[u8]| {
    // Don't panic on empty data
    if data.is_empty() {
        return;
    }

    // Try to parse as IPLD DAG-CBOR
    if let Ok(ipld) = Ipld::from_dag_cbor(data) {
        // Try to serialize it back
        if let Ok(serialized) = ipld.to_dag_cbor() {
            // Try to parse the serialized version
            if let Ok(reparsed) = Ipld::from_dag_cbor(&serialized) {
                // Both should produce the same serialization
                let serialized2 = reparsed.to_dag_cbor().unwrap();
                assert_eq!(serialized, serialized2);
            }
        }
    }
});
