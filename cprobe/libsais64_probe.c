#include "../libsais/src/libsais64.c"

int64_t probe_public_libsais64(
    const uint8_t * T,
    int64_t * SA,
    int64_t n,
    int64_t fs
) {
    return libsais64(T, SA, n, fs, NULL);
}

int64_t probe_public_libsais64_freq(
    const uint8_t * T,
    int64_t * SA,
    int64_t n,
    int64_t fs,
    int64_t * freq
) {
    return libsais64(T, SA, n, fs, freq);
}

int64_t probe_public_libsais64_omp_freq(
    const uint8_t * T,
    int64_t * SA,
    int64_t n,
    int64_t fs,
    int64_t * freq,
    int64_t threads
) {
    return libsais64_omp(T, SA, n, fs, freq, threads);
}

int64_t probe_public_libsais64_gsa(
    const uint8_t * T,
    int64_t * SA,
    int64_t n,
    int64_t fs
) {
    return libsais64_gsa(T, SA, n, fs, NULL);
}

int64_t probe_public_libsais64_gsa_freq(
    const uint8_t * T,
    int64_t * SA,
    int64_t n,
    int64_t fs,
    int64_t * freq
) {
    return libsais64_gsa(T, SA, n, fs, freq);
}

int64_t probe_public_libsais64_long(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t fs
) {
    return libsais64_long(T, SA, n, k, fs);
}

int64_t probe_public_libsais64_bwt(
    const uint8_t * T,
    uint8_t * U,
    int64_t * A,
    int64_t n,
    int64_t fs
) {
    return libsais64_bwt(T, U, A, n, fs, NULL);
}

int64_t probe_public_libsais64_bwt_freq(
    const uint8_t * T,
    uint8_t * U,
    int64_t * A,
    int64_t n,
    int64_t fs,
    int64_t * freq
) {
    return libsais64_bwt(T, U, A, n, fs, freq);
}

int64_t probe_public_libsais64_bwt_aux(
    const uint8_t * T,
    uint8_t * U,
    int64_t * A,
    int64_t n,
    int64_t fs,
    int64_t r,
    int64_t * I
) {
    return libsais64_bwt_aux(T, U, A, n, fs, NULL, r, I);
}

int64_t probe_public_libsais64_bwt_aux_freq(
    const uint8_t * T,
    uint8_t * U,
    int64_t * A,
    int64_t n,
    int64_t fs,
    int64_t * freq,
    int64_t r,
    int64_t * I
) {
    return libsais64_bwt_aux(T, U, A, n, fs, freq, r, I);
}

int64_t probe_public_libsais64_unbwt(
    const uint8_t * T,
    uint8_t * U,
    int64_t * A,
    int64_t n,
    int64_t i
) {
    return libsais64_unbwt(T, U, A, n, NULL, i);
}

int64_t probe_public_libsais64_unbwt_freq(
    const uint8_t * T,
    uint8_t * U,
    int64_t * A,
    int64_t n,
    const int64_t * freq,
    int64_t i
) {
    return libsais64_unbwt(T, U, A, n, freq, i);
}

int64_t probe_public_libsais64_unbwt_aux(
    const uint8_t * T,
    uint8_t * U,
    int64_t * A,
    int64_t n,
    int64_t r,
    const int64_t * I
) {
    return libsais64_unbwt_aux(T, U, A, n, NULL, r, I);
}

int64_t probe_public_libsais64_unbwt_aux_freq(
    const uint8_t * T,
    uint8_t * U,
    int64_t * A,
    int64_t n,
    const int64_t * freq,
    int64_t r,
    const int64_t * I
) {
    return libsais64_unbwt_aux(T, U, A, n, freq, r, I);
}

int64_t probe_public_libsais64_plcp(
    const uint8_t * T,
    const int64_t * SA,
    int64_t * PLCP,
    int64_t n
) {
    return libsais64_plcp(T, SA, PLCP, n);
}

int64_t probe_public_libsais64_plcp_gsa(
    const uint8_t * T,
    const int64_t * SA,
    int64_t * PLCP,
    int64_t n
) {
    return libsais64_plcp_gsa(T, SA, PLCP, n);
}

int64_t probe_public_libsais64_lcp(
    const int64_t * PLCP,
    const int64_t * SA,
    int64_t * LCP,
    int64_t n
) {
    return libsais64_lcp(PLCP, SA, LCP, n);
}

int64_t probe_libsais64_renumber_lms_suffixes_8u(
    int64_t * SA,
    int64_t m,
    int64_t name,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais64_renumber_lms_suffixes_8u(SA, m, name, omp_block_start, omp_block_size);
}

int64_t probe_libsais64_gather_marked_lms_suffixes(
    int64_t * SA,
    int64_t m,
    int64_t l,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais64_gather_marked_lms_suffixes(SA, m, l, omp_block_start, omp_block_size);
}

int64_t probe_libsais64_renumber_and_gather_lms_suffixes_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t fs,
    int64_t threads
) {
    return libsais64_renumber_and_gather_lms_suffixes_omp(SA, n, m, fs, threads, NULL);
}

int64_t probe_libsais64_renumber_distinct_lms_suffixes_32s_4k(
    int64_t * SA,
    int64_t m,
    int64_t name,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais64_renumber_distinct_lms_suffixes_32s_4k(SA, m, name, omp_block_start, omp_block_size);
}

int64_t probe_libsais64_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t threads
) {
    return libsais64_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(SA, n, m, threads, NULL);
}

int64_t probe_libsais64_renumber_unique_and_nonunique_lms_suffixes_32s(
    int64_t * T,
    int64_t * SA,
    int64_t m,
    int64_t f,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais64_renumber_unique_and_nonunique_lms_suffixes_32s(T, SA, m, f, omp_block_start, omp_block_size);
}

int64_t probe_libsais64_renumber_unique_and_nonunique_lms_suffixes_32s_omp(
    int64_t * T,
    int64_t * SA,
    int64_t m,
    int64_t threads
) {
    return libsais64_renumber_unique_and_nonunique_lms_suffixes_32s_omp(T, SA, m, threads, NULL);
}
