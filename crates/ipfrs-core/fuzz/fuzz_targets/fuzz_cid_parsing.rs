//! Fuzz test for CID parsing
//!
//! Tests robustness of CID::from_str() against arbitrary string inputs

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::str;

fuzz_target!(|data: &[u8]| {
    // Try to parse the data as UTF-8 string
    if let Ok(s) = str::from_utf8(data) {
        // Attempt to parse as CID - should not panic
        let _ = s.parse::<ipfrs_core::Cid>();

        // If parsing succeeds, verify round-trip
        if let Ok(cid) = s.parse::<ipfrs_core::Cid>() {
            let cid_str = cid.to_string();
            // Parsing the string representation should succeed
            let cid2 = cid_str.parse::<ipfrs_core::Cid>().expect("CID round-trip failed");
            assert_eq!(cid, cid2);
        }
    }
});
