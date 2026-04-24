#include "../libsais/src/libsais.c"

sa_sint_t probe_renumber_lms_suffixes_8u(
    sa_sint_t * SA,
    sa_sint_t m,
    sa_sint_t name,
    fast_sint_t omp_block_start,
    fast_sint_t omp_block_size
) {
    return libsais_renumber_lms_suffixes_8u(SA, m, name, omp_block_start, omp_block_size);
}

fast_sint_t probe_gather_marked_lms_suffixes(
    sa_sint_t * SA,
    sa_sint_t m,
    fast_sint_t l,
    fast_sint_t omp_block_start,
    fast_sint_t omp_block_size
) {
    return libsais_gather_marked_lms_suffixes(SA, m, l, omp_block_start, omp_block_size);
}

sa_sint_t probe_renumber_distinct_lms_suffixes_32s_4k(
    sa_sint_t * SA,
    sa_sint_t m,
    sa_sint_t name,
    fast_sint_t omp_block_start,
    fast_sint_t omp_block_size
) {
    return libsais_renumber_distinct_lms_suffixes_32s_4k(SA, m, name, omp_block_start, omp_block_size);
}

sa_sint_t probe_renumber_unique_and_nonunique_lms_suffixes_32s(
    sa_sint_t * T,
    sa_sint_t * SA,
    sa_sint_t m,
    sa_sint_t f,
    fast_sint_t omp_block_start,
    fast_sint_t omp_block_size
) {
    return libsais_renumber_unique_and_nonunique_lms_suffixes_32s(T, SA, m, f, omp_block_start, omp_block_size);
}

sa_sint_t probe_renumber_unique_and_nonunique_lms_suffixes_32s_omp(
    sa_sint_t * T,
    sa_sint_t * SA,
    sa_sint_t m,
    sa_sint_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais_alloc_thread_state(threads) : NULL;
    sa_sint_t result = libsais_renumber_unique_and_nonunique_lms_suffixes_32s_omp(T, SA, m, threads, thread_state);
    libsais_free_thread_state(thread_state);
    return result;
}

sa_sint_t probe_renumber_and_gather_lms_suffixes_omp(
    sa_sint_t * SA,
    sa_sint_t n,
    sa_sint_t m,
    sa_sint_t fs,
    sa_sint_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais_alloc_thread_state(threads) : NULL;
    sa_sint_t result = libsais_renumber_and_gather_lms_suffixes_omp(SA, n, m, fs, threads, thread_state);
    libsais_free_thread_state(thread_state);
    return result;
}

sa_sint_t probe_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
    sa_sint_t * SA,
    sa_sint_t n,
    sa_sint_t m,
    sa_sint_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais_alloc_thread_state(threads) : NULL;
    sa_sint_t result = libsais_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(SA, n, m, threads, thread_state);
    libsais_free_thread_state(thread_state);
    return result;
}

sa_sint_t probe_main_32s_entry(
    sa_sint_t * T,
    sa_sint_t * SA,
    sa_sint_t n,
    sa_sint_t k,
    sa_sint_t fs,
    sa_sint_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais_alloc_thread_state(threads) : NULL;
    sa_sint_t result = libsais_main_32s_entry(T, SA, n, k, fs, threads, thread_state);
    libsais_free_thread_state(thread_state);
    return result;
}

sa_sint_t probe_public_libsais(
    const uint8_t * T,
    sa_sint_t * SA,
    sa_sint_t n,
    sa_sint_t fs
) {
    return libsais(T, SA, n, fs, NULL);
}
