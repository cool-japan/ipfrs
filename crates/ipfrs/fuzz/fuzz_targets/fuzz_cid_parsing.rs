#![no_main]

use libfuzzer_sys::fuzz_target;
use std::str::FromStr;

fuzz_target!(|data: &[u8]| {
    // Convert to string
    if let Ok(s) = std::str::from_utf8(data) {
        // Try to parse as CID - should not panic
        let _ = ipfrs::Cid::from_str(s);

        // Try to parse with base encoding
        if let Ok(cid) = ipfrs::Cid::from_str(s) {
            // Verify round-trip
            let cid_string = cid.to_string();
            if let Ok(reparsed) = ipfrs::Cid::from_str(&cid_string) {
                assert_eq!(cid, reparsed);
            }
        }
    }
});
