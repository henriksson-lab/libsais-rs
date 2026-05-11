fn main() {
    println!("cargo:rerun-if-changed=build.rs");
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

    cc::Build::new()
        .file("cprobe/libsais_probe.c")
        .file("cprobe/libsais64_probe.c")
        .file("cprobe/libsais16_probe.c")
        .file("cprobe/libsais16x64_probe.c")
        .include("libsais/include")
        .flag_if_supported("-std=c99")
        .compile("libsais_probe");
}
