fn main() -> Result<(), Box<dyn std::error::Error>> {
    // When the `grpc` Cargo feature is enabled, use tonic-prost-build to emit
    // both message types *and* the service-server/client stubs consumed by
    // `grpc.rs` via `tonic::include_proto!()`.
    //
    // Without the feature, fall back to prost-build (message types only) so
    // that the default (pure-Rust) build stays lean and does not require
    // `protoc` to be installed for ordinary development.
    let grpc_feature = std::env::var("CARGO_FEATURE_GRPC").is_ok();

    if grpc_feature {
        // tonic-build 0.14 moved compile_protos to tonic-prost-build.
        tonic_prost_build::configure()
            .build_server(true)
            .build_client(true)
            // proto3 `optional` fields only became non-experimental in protoc
            // 3.15.0; older toolchains (e.g. 3.12.x) reject them without this
            // flag. Mirrors the prost-build fallback below.
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
    } else {
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
    }

    // When the `python` feature is active, `ipfrs-interface` embeds the Python
    // interpreter via PyO3's `auto-initialize`, which requires libpython to be
    // linked. During a workspace `--all-features` build, feature unification with
    // the `ipfrs-python` crate (which enables PyO3's `extension-module`) turns off
    // PyO3's automatic libpython link directives, so we re-add them here.
    //
    // Derive the library directory and name from pyo3-build-config, which honors
    // `PYO3_PYTHON` and resolves the correct values on macOS frameworks, Linux and
    // Windows. Never hardcode a Python version or an absolute path here.
    if std::env::var("CARGO_FEATURE_PYTHON").is_ok() {
        let config = pyo3_build_config::get();
        if let Some(lib_dir) = config.lib_dir() {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
        if let Some(lib_name) = config.lib_name() {
            println!("cargo:rustc-link-lib={lib_name}");
        }
    }

    Ok(())
}
