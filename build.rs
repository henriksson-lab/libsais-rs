fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "upstream-c")]
    build_upstream_c();
}

#[cfg(feature = "upstream-c")]
fn build_upstream_c() {
    println!("cargo:rerun-if-changed=cprobe/libsais_probe.c");
    println!("cargo:rerun-if-changed=cprobe/libsais64_probe.c");
    println!("cargo:rerun-if-changed=cprobe/libsais16_probe.c");
    println!("cargo:rerun-if-changed=cprobe/libsais16x64_probe.c");
    println!("cargo:rerun-if-changed=libsais/src/libsais.c");
    println!("cargo:rerun-if-changed=libsais/src/libsais64.c");
    println!("cargo:rerun-if-changed=libsais/src/libsais16.c");
    println!("cargo:rerun-if-changed=libsais/src/libsais16x64.c");
    println!("cargo:rerun-if-changed=libsais/include/libsais.h");
    println!("cargo:rerun-if-changed=libsais/include/libsais64.h");
    println!("cargo:rerun-if-changed=libsais/include/libsais16.h");
    println!("cargo:rerun-if-changed=libsais/include/libsais16x64.h");
    let mut build = cc::Build::new();
    build
        .file("cprobe/libsais_probe.c")
        .file("cprobe/libsais64_probe.c")
        .file("cprobe/libsais16_probe.c")
        .file("cprobe/libsais16x64_probe.c")
        .include("libsais/include")
        .define("LIBSAIS_OPENMP", None)
        .flag_if_supported("-std=c99")
        .flag_if_supported("-O3")
        .flag_if_supported("-march=native");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // Apple clang ships no libgomp and rejects a bare -fopenmp. Use the LLVM
        // OpenMP runtime instead, from LIBOMP_PREFIX or Homebrew.
        let prefix = std::env::var("LIBOMP_PREFIX").ok().unwrap_or_else(|| {
            let out = std::process::Command::new("brew")
                .args(["--prefix", "libomp"])
                .output()
                .expect("upstream-c on macOS needs libomp: brew install libomp");
            assert!(
                out.status.success(),
                "upstream-c on macOS needs libomp: brew install libomp"
            );
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        });
        build
            .flag("-Xpreprocessor")
            .flag("-fopenmp")
            .include(format!("{prefix}/include"));
        println!("cargo:rustc-link-search=native={prefix}/lib");
        println!("cargo:rustc-link-lib=omp");
    } else {
        build.flag("-fopenmp");
        println!("cargo:rustc-link-lib=gomp");
    }

    build.compile("libsais_probe");
}
