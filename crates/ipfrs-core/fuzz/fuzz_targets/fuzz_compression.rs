#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::{compress, decompress, compression_ratio, CompressionAlgorithm};
use bytes::Bytes;

fuzz_target!(|data: &[u8]| {
    // Test with arbitrary data
    let bytes_data = Bytes::copy_from_slice(data);

    // Test all compression algorithms
    for algorithm in CompressionAlgorithm::all() {
        // Test various compression levels
        for level in [0, 3, 5, 9] {
            // Compress should not panic
            if let Ok(compressed) = compress(&bytes_data, *algorithm, level) {
                // Decompress should not panic and should match original
                if let Ok(decompressed) = decompress(&compressed, *algorithm) {
                    assert_eq!(bytes_data, decompressed, "Roundtrip failed for {:?}", algorithm);
                }

                // Compression ratio should be valid
                if let Ok(ratio) = compression_ratio(&bytes_data, *algorithm, level) {
                    assert!(ratio >= 0.0, "Invalid compression ratio");
                    assert!(ratio.is_finite(), "Compression ratio is not finite");
                }
            }
        }

        // Test None algorithm specifically (should always succeed)
        if *algorithm == CompressionAlgorithm::None {
            let compressed = compress(&bytes_data, *algorithm, 5).expect("None compression failed");
            assert_eq!(bytes_data, compressed, "None algorithm should be identity");
        }
    }

    // Test invalid compression levels (should return error, not panic)
    for invalid_level in [10, 50, 100, 255] {
        let _ = compress(&bytes_data, CompressionAlgorithm::Zstd, invalid_level);
    }

    // Test decompressing random data (should handle gracefully)
    let _ = decompress(&bytes_data, CompressionAlgorithm::Zstd);
    let _ = decompress(&bytes_data, CompressionAlgorithm::Lz4);
});
