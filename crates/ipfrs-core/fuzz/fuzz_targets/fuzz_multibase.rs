#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::{parse_cid_with_base, MultibaseEncoding, CidExt};

fuzz_target!(|data: &[u8]| {
    // Fuzz multibase parsing
    if let Ok(s) = std::str::from_utf8(data) {
        // Try to parse as CID with multibase
        let _ = parse_cid_with_base(s);
    }

    // Fuzz multibase encoding round-trip
    if data.len() > 2 && data.len() < 1000 {
        let encodings = [
            MultibaseEncoding::Base32Lower,
            MultibaseEncoding::Base32Upper,
            MultibaseEncoding::Base58Btc,
            MultibaseEncoding::Base64,
            MultibaseEncoding::Base64Url,
        ];

        for encoding in &encodings {
            // Try encoding arbitrary data
            if let Ok(cid) = ipfrs_core::CidBuilder::new().build(data) {
                let encoded = cid.to_string_with_base(*encoding);
                // Try parsing it back
                let _ = parse_cid_with_base(&encoded);
            }
        }
    }
});
