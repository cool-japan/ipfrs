//! Standalone Arrow IPC helpers for gradient serialization.
//!
//! These functions serialize/deserialize raw `f32` gradient slices to/from
//! Apache Arrow IPC format so that gradients can be stored as content-addressed
//! blocks in the IPFS block store.

/// Serialize a gradient tensor as an Arrow IPC RecordBatch.
///
/// The output bytes contain a single record batch with two columns:
/// - `"index"` (`Int32`): element indices 0..N
/// - `"value"` (`Float32`): gradient values
///
/// This function is a standalone companion to
/// `ComputationGraphStore::store_gradient_as_arrow`.  It does **not**
/// require a node id or a graph — it operates purely on the raw gradient
/// slice and is suitable for use in `DistributedGradientAccumulator`.
pub fn store_gradient_as_arrow(gradient: &[f32]) -> anyhow::Result<Vec<u8>> {
    use arrow::array::{Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::FileWriter;
    use arrow::record_batch::RecordBatch;
    use std::io::Cursor;
    use std::sync::Arc;

    let n = gradient.len();
    let indices: Int32Array = (0i32..(n as i32)).collect();
    let values: Float32Array = gradient.iter().copied().collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("index", DataType::Int32, false),
        Field::new("value", DataType::Float32, false),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(indices), Arc::new(values)])?;

    let mut buf = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut writer = FileWriter::try_new(cursor, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
    }

    Ok(buf)
}

/// Deserialize gradient from Arrow IPC bytes produced by [`store_gradient_as_arrow`].
///
/// Reads the `"value"` column and returns it as `Vec<f32>`.
pub fn load_gradient_from_arrow(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    use arrow::array::Float32Array;
    use arrow::ipc::reader::FileReader;
    use std::io::Cursor;

    let cursor = Cursor::new(bytes);
    let mut reader = FileReader::try_new(cursor, None)?;

    let mut values: Vec<f32> = Vec::new();

    for batch_result in &mut reader {
        let batch = batch_result?;
        // Locate the "value" column
        let schema = batch.schema();
        let col_idx = schema
            .index_of("value")
            .map_err(|_| anyhow::anyhow!("Arrow IPC block missing 'value' column"))?;

        let col = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow::anyhow!("'value' column is not Float32"))?;

        values.extend_from_slice(col.values());
    }

    Ok(values)
}

#[cfg(test)]
mod arrow_ipc_standalone_tests {
    use super::*;

    #[test]
    fn test_arrow_ipc_roundtrip() {
        // Generate 1000 f32 values in a deterministic pattern.
        let original: Vec<f32> = (0u32..1000).map(|i| (i as f32) * 0.001).collect();

        let bytes =
            store_gradient_as_arrow(&original).expect("store_gradient_as_arrow should succeed");
        assert!(!bytes.is_empty(), "IPC bytes must not be empty");

        let loaded =
            load_gradient_from_arrow(&bytes).expect("load_gradient_from_arrow should succeed");

        assert_eq!(loaded.len(), original.len(), "element count must match");
        for (i, (&orig, &val)) in original.iter().zip(loaded.iter()).enumerate() {
            assert!(
                (orig - val).abs() < 1e-6,
                "value mismatch at index {i}: orig={orig}, loaded={val}"
            );
        }
    }

    #[test]
    fn test_arrow_ipc_empty() {
        let empty: Vec<f32> = vec![];

        let bytes = store_gradient_as_arrow(&empty)
            .expect("store_gradient_as_arrow on empty slice should succeed");

        let loaded = load_gradient_from_arrow(&bytes)
            .expect("load_gradient_from_arrow on empty IPC should succeed");

        assert!(loaded.is_empty(), "loaded gradient must also be empty");
    }
}
