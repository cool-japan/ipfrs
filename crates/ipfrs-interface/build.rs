fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile all proto files using prost-build
    // NOTE: tonic-build 0.14 changed API significantly
    // For now, using prost_build directly without tonic service generation
    // TODO: Add proper tonic service generation when tonic-build API is stable
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "proto/block.proto",
                "proto/dag.proto",
                "proto/file.proto",
                "proto/tensor.proto",
            ],
            &["proto"],
        )?;
    Ok(())
}
