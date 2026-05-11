#include "../libsais/src/libsais16x64.c"

int64_t probe_public_libsais16x64(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t fs
) {
    return libsais16x64(T, SA, n, fs, NULL);
}

int64_t probe_public_libsais16x64_freq(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t fs,
    int64_t * freq
) {
    return libsais16x64(T, SA, n, fs, freq);
}

int64_t probe_public_libsais16x64_gsa(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t fs
) {
    return libsais16x64_gsa(T, SA, n, fs, NULL);
}

int64_t probe_public_libsais16x64_gsa_freq(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t fs,
    int64_t * freq
) {
    return libsais16x64_gsa(T, SA, n, fs, freq);
}

int64_t probe_public_libsais16x64_long(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t fs
) {
    return libsais16x64_long(T, SA, n, k, fs);
}

void probe_libsais16x64_final_sorting_scan_left_to_right_32s(
    const int64_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_final_sorting_scan_left_to_right_32s(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_final_sorting_scan_right_to_left_32s(
    const int64_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_final_sorting_scan_right_to_left_32s(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_clear_lms_suffixes_omp(
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * bucket_start,
    int64_t * bucket_end,
    int64_t threads
) {
    libsais16x64_clear_lms_suffixes_omp(SA, n, k, bucket_start, bucket_end, threads);
}

void probe_libsais16x64_flip_suffix_markers_omp(
    int64_t * SA,
    int64_t l,
    int64_t threads
) {
    libsais16x64_flip_suffix_markers_omp(SA, l, threads);
}

void probe_libsais16x64_induce_final_order_32s_6k(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16x64_induce_final_order_32s_6k(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16x64_free_thread_state(thread_state);
}

void probe_libsais16x64_induce_final_order_32s_4k(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16x64_induce_final_order_32s_4k(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16x64_free_thread_state(thread_state);
}

void probe_libsais16x64_induce_final_order_32s_2k(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16x64_induce_final_order_32s_2k(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16x64_free_thread_state(thread_state);
}

void probe_libsais16x64_induce_final_order_32s_1k(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16x64_induce_final_order_32s_1k(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16x64_free_thread_state(thread_state);
}

void probe_libsais16x64_induce_partial_order_32s_6k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t first_lms_suffix,
    int64_t left_suffixes_count,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16x64_induce_partial_order_32s_6k_omp(T, SA, n, k, buckets, first_lms_suffix, left_suffixes_count, threads, thread_state);
    }
    libsais16x64_free_thread_state(thread_state);
}

void probe_libsais16x64_induce_partial_order_32s_4k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16x64_induce_partial_order_32s_4k_omp(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16x64_free_thread_state(thread_state);
}

void probe_libsais16x64_induce_partial_order_32s_2k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16x64_induce_partial_order_32s_2k_omp(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16x64_free_thread_state(thread_state);
}

void probe_libsais16x64_induce_partial_order_32s_1k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16x64_induce_partial_order_32s_1k_omp(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16x64_free_thread_state(thread_state);
}

void probe_libsais16x64_induce_partial_order_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t flags,
    int64_t * buckets,
    int64_t first_lms_suffix,
    int64_t left_suffixes_count,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16x64_induce_partial_order_16u_omp(T, SA, n, k, flags, buckets, first_lms_suffix, left_suffixes_count, threads, thread_state);
    }
    libsais16x64_free_thread_state(thread_state);
}

int64_t probe_libsais16x64_induce_final_order_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t flags,
    int64_t r,
    int64_t * I,
    int64_t * buckets,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    int64_t result = (thread_state != NULL || threads == 1)
        ? libsais16x64_induce_final_order_16u_omp(T, SA, n, k, flags, r, I, buckets, threads, thread_state)
        : -2;
    libsais16x64_free_thread_state(thread_state);
    return result;
}

int64_t probe_public_libsais16x64_bwt(
    const uint16_t * T,
    uint16_t * U,
    int64_t * A,
    int64_t n,
    int64_t fs
) {
    return libsais16x64_bwt(T, U, A, n, fs, NULL);
}

int64_t probe_public_libsais16x64_bwt_freq(
    const uint16_t * T,
    uint16_t * U,
    int64_t * A,
    int64_t n,
    int64_t fs,
    int64_t * freq
) {
    return libsais16x64_bwt(T, U, A, n, fs, freq);
}

int64_t probe_public_libsais16x64_bwt_aux(
    const uint16_t * T,
    uint16_t * U,
    int64_t * A,
    int64_t n,
    int64_t fs,
    int64_t r,
    int64_t * I
) {
    return libsais16x64_bwt_aux(T, U, A, n, fs, NULL, r, I);
}

int64_t probe_public_libsais16x64_bwt_aux_freq(
    const uint16_t * T,
    uint16_t * U,
    int64_t * A,
    int64_t n,
    int64_t fs,
    int64_t * freq,
    int64_t r,
    int64_t * I
) {
    return libsais16x64_bwt_aux(T, U, A, n, fs, freq, r, I);
}

int64_t probe_public_libsais16x64_unbwt(
    const uint16_t * T,
    uint16_t * U,
    int64_t * A,
    int64_t n,
    int64_t i
) {
    return libsais16x64_unbwt(T, U, A, n, NULL, i);
}

int64_t probe_public_libsais16x64_unbwt_freq(
    const uint16_t * T,
    uint16_t * U,
    int64_t * A,
    int64_t n,
    const int64_t * freq,
    int64_t i
) {
    return libsais16x64_unbwt(T, U, A, n, freq, i);
}

int64_t probe_public_libsais16x64_unbwt_aux(
    const uint16_t * T,
    uint16_t * U,
    int64_t * A,
    int64_t n,
    int64_t r,
    const int64_t * I
) {
    return libsais16x64_unbwt_aux(T, U, A, n, NULL, r, I);
}

int64_t probe_public_libsais16x64_unbwt_aux_freq(
    const uint16_t * T,
    uint16_t * U,
    int64_t * A,
    int64_t n,
    const int64_t * freq,
    int64_t r,
    const int64_t * I
) {
    return libsais16x64_unbwt_aux(T, U, A, n, freq, r, I);
}

int64_t probe_public_libsais16x64_plcp(
    const uint16_t * T,
    const int64_t * SA,
    int64_t * PLCP,
    int64_t n
) {
    return libsais16x64_plcp(T, SA, PLCP, n);
}

int64_t probe_public_libsais16x64_plcp_gsa(
    const uint16_t * T,
    const int64_t * SA,
    int64_t * PLCP,
    int64_t n
) {
    return libsais16x64_plcp_gsa(T, SA, PLCP, n);
}

int64_t probe_public_libsais16x64_lcp(
    const int64_t * PLCP,
    const int64_t * SA,
    int64_t * LCP,
    int64_t n
) {
    return libsais16x64_lcp(PLCP, SA, LCP, n);
}

void probe_libsais16x64_gather_lms_suffixes_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_gather_lms_suffixes_16u(T, SA, n, m, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_count_and_gather_lms_suffixes_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t * buckets,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_count_and_gather_lms_suffixes_16u(T, SA, n, buckets, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_initialize_buckets_start_and_end_16u(
    int64_t * buckets,
    int64_t * freq
) {
    return libsais16x64_initialize_buckets_start_and_end_16u(buckets, freq);
}

int64_t probe_libsais16x64_initialize_buckets_for_lms_suffixes_radix_sort_16u(
    const uint16_t * T,
    int64_t * buckets,
    int64_t first_lms_suffix
) {
    return libsais16x64_initialize_buckets_for_lms_suffixes_radix_sort_16u(T, buckets, first_lms_suffix);
}

void probe_libsais16x64_radix_sort_lms_suffixes_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_radix_sort_lms_suffixes_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_initialize_buckets_for_partial_sorting_16u(
    const uint16_t * T,
    int64_t * buckets,
    int64_t first_lms_suffix,
    int64_t left_suffixes_count
) {
    libsais16x64_initialize_buckets_for_partial_sorting_16u(T, buckets, first_lms_suffix, left_suffixes_count);
}

int64_t probe_libsais16x64_partial_sorting_scan_left_to_right_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t * buckets,
    int64_t d,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_partial_sorting_scan_left_to_right_16u(T, SA, buckets, d, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_partial_sorting_scan_right_to_left_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t * buckets,
    int64_t d,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_partial_sorting_scan_right_to_left_16u(T, SA, buckets, d, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_partial_gsa_scan_right_to_left_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t * buckets,
    int64_t d,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_partial_gsa_scan_right_to_left_16u(T, SA, buckets, d, omp_block_start, omp_block_size);
}

void probe_libsais16x64_partial_sorting_shift_markers_16u_omp(
    int64_t * SA,
    int64_t n,
    const int64_t * buckets,
    int64_t threads
) {
    libsais16x64_partial_sorting_shift_markers_16u_omp(SA, n, buckets, threads);
}

void probe_libsais16x64_final_sorting_scan_left_to_right_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_final_sorting_scan_left_to_right_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_final_sorting_scan_right_to_left_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_final_sorting_scan_right_to_left_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_final_gsa_scan_right_to_left_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_final_gsa_scan_right_to_left_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_final_bwt_scan_left_to_right_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_final_bwt_scan_left_to_right_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_final_bwt_scan_right_to_left_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_final_bwt_scan_right_to_left_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_final_bwt_aux_scan_left_to_right_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t rm,
    int64_t * I,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_final_bwt_aux_scan_left_to_right_16u(T, SA, rm, I, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_final_bwt_aux_scan_right_to_left_16u(
    const uint16_t * T,
    int64_t * SA,
    int64_t rm,
    int64_t * I,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_final_bwt_aux_scan_right_to_left_16u(T, SA, rm, I, induction_bucket, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_renumber_lms_suffixes_16u(
    int64_t * SA,
    int64_t m,
    int64_t name,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_renumber_lms_suffixes_16u(SA, m, name, omp_block_start, omp_block_size);
}

void probe_libsais16x64_place_lms_suffixes_interval_16u(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t flags,
    int64_t * buckets
) {
    libsais16x64_place_lms_suffixes_interval_16u(SA, n, m, flags, buckets);
}

void probe_libsais16x64_bwt_copy_16u(
    uint16_t * U,
    int64_t * A,
    int64_t n
) {
    libsais16x64_bwt_copy_16u(U, A, n);
}

void probe_libsais16x64_gather_lms_suffixes_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t threads
) {
    libsais16x64_gather_lms_suffixes_16u_omp(T, SA, n, threads, NULL);
}

int64_t probe_libsais16x64_count_and_gather_lms_suffixes_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t * buckets,
    int64_t threads
) {
    return libsais16x64_count_and_gather_lms_suffixes_16u_omp(T, SA, n, buckets, threads, NULL);
}

void probe_libsais16x64_radix_sort_lms_suffixes_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t flags,
    int64_t * buckets,
    int64_t threads
) {
    libsais16x64_radix_sort_lms_suffixes_16u_omp(T, SA, n, m, flags, buckets, threads, NULL);
}

int64_t probe_libsais16x64_partial_sorting_scan_left_to_right_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t left_suffixes_count,
    int64_t d,
    int64_t threads
) {
    return libsais16x64_partial_sorting_scan_left_to_right_16u_omp(T, SA, n, k, buckets, left_suffixes_count, d, threads, NULL);
}

void probe_libsais16x64_partial_sorting_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t first_lms_suffix,
    int64_t left_suffixes_count,
    int64_t d,
    int64_t threads
) {
    libsais16x64_partial_sorting_scan_right_to_left_16u_omp(T, SA, n, k, buckets, first_lms_suffix, left_suffixes_count, d, threads, NULL);
}

void probe_libsais16x64_partial_gsa_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t first_lms_suffix,
    int64_t left_suffixes_count,
    int64_t d,
    int64_t threads
) {
    libsais16x64_partial_gsa_scan_right_to_left_16u_omp(T, SA, n, k, buckets, first_lms_suffix, left_suffixes_count, d, threads, NULL);
}

int64_t probe_libsais16x64_renumber_lms_suffixes_16u_omp(
    int64_t * SA,
    int64_t m,
    int64_t threads
) {
    return libsais16x64_renumber_lms_suffixes_16u_omp(SA, m, threads, NULL);
}

void probe_libsais16x64_final_bwt_scan_left_to_right_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_final_bwt_scan_left_to_right_16u_omp(T, SA, n, k, induction_bucket, threads, NULL);
}

void probe_libsais16x64_final_bwt_aux_scan_left_to_right_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t rm,
    int64_t * I,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_final_bwt_aux_scan_left_to_right_16u_omp(T, SA, n, k, rm, I, induction_bucket, threads, NULL);
}

void probe_libsais16x64_final_sorting_scan_left_to_right_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_final_sorting_scan_left_to_right_16u_omp(T, SA, n, k, induction_bucket, threads, NULL);
}

int64_t probe_libsais16x64_final_bwt_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * induction_bucket,
    int64_t threads
) {
    return libsais16x64_final_bwt_scan_right_to_left_16u_omp(T, SA, n, k, induction_bucket, threads, NULL);
}

void probe_libsais16x64_final_bwt_aux_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t rm,
    int64_t * I,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_final_bwt_aux_scan_right_to_left_16u_omp(T, SA, n, k, rm, I, induction_bucket, threads, NULL);
}

void probe_libsais16x64_final_sorting_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t omp_block_start,
    int64_t omp_block_size,
    int64_t k,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_final_sorting_scan_right_to_left_16u_omp(T, SA, omp_block_start, omp_block_size, k, induction_bucket, threads, NULL);
}

void probe_libsais16x64_final_gsa_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int64_t * SA,
    int64_t omp_block_start,
    int64_t omp_block_size,
    int64_t k,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_final_gsa_scan_right_to_left_16u_omp(T, SA, omp_block_start, omp_block_size, k, induction_bucket, threads, NULL);
}

void probe_libsais16x64_bwt_copy_16u_omp(
    uint16_t * U,
    int64_t * A,
    int64_t n,
    int64_t threads
) {
    libsais16x64_bwt_copy_16u_omp(U, A, n, threads);
}

int64_t probe_libsais16x64_gather_marked_lms_suffixes(
    int64_t * SA,
    int64_t m,
    int64_t l,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_gather_marked_lms_suffixes(SA, m, l, omp_block_start, omp_block_size);
}

void probe_libsais16x64_gather_marked_lms_suffixes_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t fs,
    int64_t threads
) {
    libsais16x64_gather_marked_lms_suffixes_omp(SA, n, m, fs, threads, NULL);
}

int64_t probe_libsais16x64_renumber_and_gather_lms_suffixes_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t fs,
    int64_t threads
) {
    return libsais16x64_renumber_and_gather_lms_suffixes_omp(SA, n, m, fs, threads, NULL);
}

void probe_libsais16x64_reconstruct_lms_suffixes(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_reconstruct_lms_suffixes(SA, n, m, omp_block_start, omp_block_size);
}

void probe_libsais16x64_reconstruct_lms_suffixes_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t threads
) {
    libsais16x64_reconstruct_lms_suffixes_omp(SA, n, m, threads);
}

int64_t probe_libsais16x64_renumber_distinct_lms_suffixes_32s_4k(
    int64_t * SA,
    int64_t m,
    int64_t name,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_renumber_distinct_lms_suffixes_32s_4k(SA, m, name, omp_block_start, omp_block_size);
}

void probe_libsais16x64_mark_distinct_lms_suffixes_32s(
    int64_t * SA,
    int64_t m,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_mark_distinct_lms_suffixes_32s(SA, m, omp_block_start, omp_block_size);
}

void probe_libsais16x64_clamp_lms_suffixes_length_32s(
    int64_t * SA,
    int64_t m,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_clamp_lms_suffixes_length_32s(SA, m, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_renumber_distinct_lms_suffixes_32s_4k_omp(
    int64_t * SA,
    int64_t m,
    int64_t threads
) {
    return libsais16x64_renumber_distinct_lms_suffixes_32s_4k_omp(SA, m, threads, NULL);
}

void probe_libsais16x64_mark_distinct_lms_suffixes_32s_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t threads
) {
    libsais16x64_mark_distinct_lms_suffixes_32s_omp(SA, n, m, threads);
}

void probe_libsais16x64_clamp_lms_suffixes_length_32s_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t threads
) {
    libsais16x64_clamp_lms_suffixes_length_32s_omp(SA, n, m, threads);
}

int64_t probe_libsais16x64_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t threads
) {
    return libsais16x64_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(SA, n, m, threads, NULL);
}

int64_t probe_libsais16x64_renumber_unique_and_nonunique_lms_suffixes_32s(
    int64_t * T,
    int64_t * SA,
    int64_t m,
    int64_t f,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_renumber_unique_and_nonunique_lms_suffixes_32s(T, SA, m, f, omp_block_start, omp_block_size);
}

void probe_libsais16x64_compact_unique_and_nonunique_lms_suffixes_32s(
    int64_t * SA,
    int64_t m,
    int64_t * pl,
    int64_t * pr,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    fast_sint_t l = *pl;
    fast_sint_t r = *pr;
    libsais16x64_compact_unique_and_nonunique_lms_suffixes_32s(SA, m, &l, &r, omp_block_start, omp_block_size);
    *pl = l;
    *pr = r;
}

int64_t probe_libsais16x64_renumber_unique_and_nonunique_lms_suffixes_32s_omp(
    int64_t * T,
    int64_t * SA,
    int64_t m,
    int64_t threads
) {
    return libsais16x64_renumber_unique_and_nonunique_lms_suffixes_32s_omp(T, SA, m, threads, NULL);
}

void probe_libsais16x64_compact_unique_and_nonunique_lms_suffixes_32s_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t fs,
    int64_t f,
    int64_t threads
) {
    libsais16x64_compact_unique_and_nonunique_lms_suffixes_32s_omp(SA, n, m, fs, f, threads, NULL);
}

int64_t probe_libsais16x64_compact_lms_suffixes_32s_omp(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t fs,
    int64_t threads
) {
    return libsais16x64_compact_lms_suffixes_32s_omp(T, SA, n, m, fs, threads, NULL);
}

void probe_libsais16x64_merge_unique_lms_suffixes_32s(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t l,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_merge_unique_lms_suffixes_32s(T, SA, n, m, l, omp_block_start, omp_block_size);
}

void probe_libsais16x64_merge_nonunique_lms_suffixes_32s(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t l,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_merge_nonunique_lms_suffixes_32s(SA, n, m, l, omp_block_start, omp_block_size);
}

void probe_libsais16x64_merge_unique_lms_suffixes_32s_omp(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t threads
) {
    libsais16x64_merge_unique_lms_suffixes_32s_omp(T, SA, n, m, threads, NULL);
}

void probe_libsais16x64_merge_nonunique_lms_suffixes_32s_omp(
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t f,
    int64_t threads
) {
    libsais16x64_merge_nonunique_lms_suffixes_32s_omp(SA, n, m, f, threads, NULL);
}

void probe_libsais16x64_merge_compacted_lms_suffixes_32s_omp(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t f,
    int64_t threads
) {
    libsais16x64_merge_compacted_lms_suffixes_32s_omp(T, SA, n, m, f, threads, NULL);
}

void probe_libsais16x64_radix_sort_lms_suffixes_32s_6k(
    const int64_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_radix_sort_lms_suffixes_32s_6k(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_radix_sort_lms_suffixes_32s_2k(
    const int64_t * T,
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_radix_sort_lms_suffixes_32s_2k(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_radix_sort_lms_suffixes_32s_6k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_radix_sort_lms_suffixes_32s_6k_omp(T, SA, n, m, induction_bucket, threads, NULL);
}

void probe_libsais16x64_radix_sort_lms_suffixes_32s_2k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_radix_sort_lms_suffixes_32s_2k_omp(T, SA, n, m, induction_bucket, threads, NULL);
}

int64_t probe_libsais16x64_radix_sort_lms_suffixes_32s_1k(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t * buckets
) {
    return libsais16x64_radix_sort_lms_suffixes_32s_1k(T, SA, n, buckets);
}

void probe_libsais16x64_radix_sort_set_markers_32s_6k(
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_radix_sort_set_markers_32s_6k(SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_radix_sort_set_markers_32s_4k(
    int64_t * SA,
    int64_t * induction_bucket,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_radix_sort_set_markers_32s_4k(SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16x64_radix_sort_set_markers_32s_6k_omp(
    int64_t * SA,
    int64_t k,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_radix_sort_set_markers_32s_6k_omp(SA, k, induction_bucket, threads);
}

void probe_libsais16x64_radix_sort_set_markers_32s_4k_omp(
    int64_t * SA,
    int64_t k,
    int64_t * induction_bucket,
    int64_t threads
) {
    libsais16x64_radix_sort_set_markers_32s_4k_omp(SA, k, induction_bucket, threads);
}

void probe_libsais16x64_place_lms_suffixes_histogram_32s_6k(
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t m,
    const int64_t * buckets
) {
    libsais16x64_place_lms_suffixes_histogram_32s_6k(SA, n, k, m, buckets);
}

void probe_libsais16x64_place_lms_suffixes_histogram_32s_4k(
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t m,
    const int64_t * buckets
) {
    libsais16x64_place_lms_suffixes_histogram_32s_4k(SA, n, k, m, buckets);
}

void probe_libsais16x64_place_lms_suffixes_histogram_32s_2k(
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t m,
    const int64_t * buckets
) {
    libsais16x64_place_lms_suffixes_histogram_32s_2k(SA, n, k, m, buckets);
}

int64_t probe_libsais16x64_gather_lms_suffixes_32s(
    const int64_t * T,
    int64_t * SA,
    int64_t n
) {
    return libsais16x64_gather_lms_suffixes_32s(T, SA, n);
}

int64_t probe_libsais16x64_gather_compacted_lms_suffixes_32s(
    const int64_t * T,
    int64_t * SA,
    int64_t n
) {
    return libsais16x64_gather_compacted_lms_suffixes_32s(T, SA, n);
}

void probe_libsais16x64_count_lms_suffixes_32s_2k(
    const int64_t * T,
    int64_t n,
    int64_t k,
    int64_t * buckets
) {
    libsais16x64_count_lms_suffixes_32s_2k(T, n, k, buckets);
}

int64_t probe_libsais16x64_count_and_gather_lms_suffixes_32s_4k(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_count_and_gather_lms_suffixes_32s_4k(T, SA, n, k, buckets, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_count_and_gather_lms_suffixes_32s_4k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t local_buckets,
    int64_t threads
) {
    return libsais16x64_count_and_gather_lms_suffixes_32s_4k_omp(T, SA, n, k, buckets, local_buckets, threads, NULL);
}

void probe_libsais16x64_count_suffixes_32s(
    const int64_t * T,
    int64_t n,
    int64_t k,
    int64_t * buckets
) {
    libsais16x64_count_suffixes_32s(T, n, k, buckets);
}

void probe_libsais16x64_initialize_buckets_start_and_end_32s_6k(
    int64_t k,
    int64_t * buckets
) {
    libsais16x64_initialize_buckets_start_and_end_32s_6k(k, buckets);
}

void probe_libsais16x64_initialize_buckets_start_and_end_32s_4k(
    int64_t k,
    int64_t * buckets
) {
    libsais16x64_initialize_buckets_start_and_end_32s_4k(k, buckets);
}

void probe_libsais16x64_initialize_buckets_end_32s_2k(
    int64_t k,
    int64_t * buckets
) {
    libsais16x64_initialize_buckets_end_32s_2k(k, buckets);
}

void probe_libsais16x64_initialize_buckets_start_and_end_32s_2k(
    int64_t k,
    int64_t * buckets
) {
    libsais16x64_initialize_buckets_start_and_end_32s_2k(k, buckets);
}

void probe_libsais16x64_initialize_buckets_start_32s_1k(
    int64_t k,
    int64_t * buckets
) {
    libsais16x64_initialize_buckets_start_32s_1k(k, buckets);
}

void probe_libsais16x64_initialize_buckets_end_32s_1k(
    int64_t k,
    int64_t * buckets
) {
    libsais16x64_initialize_buckets_end_32s_1k(k, buckets);
}

void probe_libsais16x64_initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(
    const int64_t * T,
    int64_t k,
    int64_t * buckets,
    int64_t first_lms_suffix
) {
    libsais16x64_initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(T, k, buckets, first_lms_suffix);
}

int64_t probe_libsais16x64_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(
    const int64_t * T,
    int64_t k,
    int64_t * buckets,
    int64_t first_lms_suffix
) {
    return libsais16x64_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(T, k, buckets, first_lms_suffix);
}

void probe_libsais16x64_initialize_buckets_for_radix_and_partial_sorting_32s_4k(
    const int64_t * T,
    int64_t k,
    int64_t * buckets,
    int64_t first_lms_suffix
) {
    libsais16x64_initialize_buckets_for_radix_and_partial_sorting_32s_4k(T, k, buckets, first_lms_suffix);
}

void probe_libsais16x64_place_lms_suffixes_interval_32s_4k(
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t m,
    const int64_t * buckets
) {
    libsais16x64_place_lms_suffixes_interval_32s_4k(SA, n, k, m, buckets);
}

void probe_libsais16x64_place_lms_suffixes_interval_32s_2k(
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t m,
    const int64_t * buckets
) {
    libsais16x64_place_lms_suffixes_interval_32s_2k(SA, n, k, m, buckets);
}

void probe_libsais16x64_place_lms_suffixes_interval_32s_1k(
    const int64_t * T,
    int64_t * SA,
    int64_t k,
    int64_t m,
    int64_t * buckets
) {
    libsais16x64_place_lms_suffixes_interval_32s_1k(T, SA, k, m, buckets);
}

int64_t probe_libsais16x64_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t threads
) {
    return libsais16x64_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(T, SA, n, m, threads);
}

void probe_libsais16x64_partial_sorting_shift_markers_32s_6k_omp(
    int64_t * SA,
    int64_t k,
    const int64_t * buckets,
    int64_t threads
) {
    libsais16x64_partial_sorting_shift_markers_32s_6k_omp(SA, k, buckets, threads);
}

void probe_libsais16x64_partial_sorting_shift_markers_32s_4k(
    int64_t * SA,
    int64_t n
) {
    libsais16x64_partial_sorting_shift_markers_32s_4k(SA, n);
}

void probe_libsais16x64_partial_sorting_shift_buckets_32s_6k(
    int64_t k,
    int64_t * buckets
) {
    libsais16x64_partial_sorting_shift_buckets_32s_6k(k, buckets);
}

int64_t probe_libsais16x64_partial_sorting_scan_left_to_right_32s_6k(
    const int64_t * T,
    int64_t * SA,
    int64_t * buckets,
    int64_t d,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_partial_sorting_scan_left_to_right_32s_6k(T, SA, buckets, d, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_partial_sorting_scan_left_to_right_32s_4k(
    const int64_t * T,
    int64_t * SA,
    int64_t k,
    int64_t * buckets,
    int64_t d,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_partial_sorting_scan_left_to_right_32s_4k(T, SA, k, buckets, d, omp_block_start, omp_block_size);
}

void probe_libsais16x64_partial_sorting_scan_left_to_right_32s_1k(
    const int64_t * T,
    int64_t * SA,
    int64_t * buckets,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_partial_sorting_scan_left_to_right_32s_1k(T, SA, buckets, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_partial_sorting_scan_left_to_right_32s_6k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t * buckets,
    int64_t left_suffixes_count,
    int64_t d,
    int64_t threads
) {
    return libsais16x64_partial_sorting_scan_left_to_right_32s_6k_omp(T, SA, n, buckets, left_suffixes_count, d, threads, NULL);
}

int64_t probe_libsais16x64_partial_sorting_scan_left_to_right_32s_4k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t d,
    int64_t threads
) {
    return libsais16x64_partial_sorting_scan_left_to_right_32s_4k_omp(T, SA, n, k, buckets, d, threads, NULL);
}

void probe_libsais16x64_partial_sorting_scan_left_to_right_32s_1k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t * buckets,
    int64_t threads
) {
    libsais16x64_partial_sorting_scan_left_to_right_32s_1k_omp(T, SA, n, buckets, threads, NULL);
}

int64_t probe_libsais16x64_partial_sorting_scan_right_to_left_32s_6k(
    const int64_t * T,
    int64_t * SA,
    int64_t * buckets,
    int64_t d,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_partial_sorting_scan_right_to_left_32s_6k(T, SA, buckets, d, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_partial_sorting_scan_right_to_left_32s_4k(
    const int64_t * T,
    int64_t * SA,
    int64_t k,
    int64_t * buckets,
    int64_t d,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_partial_sorting_scan_right_to_left_32s_4k(T, SA, k, buckets, d, omp_block_start, omp_block_size);
}

void probe_libsais16x64_partial_sorting_scan_right_to_left_32s_1k(
    const int64_t * T,
    int64_t * SA,
    int64_t * buckets,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    libsais16x64_partial_sorting_scan_right_to_left_32s_1k(T, SA, buckets, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_partial_sorting_scan_right_to_left_32s_6k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t * buckets,
    int64_t first_lms_suffix,
    int64_t left_suffixes_count,
    int64_t d,
    int64_t threads
) {
    return libsais16x64_partial_sorting_scan_right_to_left_32s_6k_omp(T, SA, n, buckets, first_lms_suffix, left_suffixes_count, d, threads, NULL);
}

int64_t probe_libsais16x64_partial_sorting_scan_right_to_left_32s_4k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t d,
    int64_t threads
) {
    return libsais16x64_partial_sorting_scan_right_to_left_32s_4k_omp(T, SA, n, k, buckets, d, threads, NULL);
}

void probe_libsais16x64_partial_sorting_scan_right_to_left_32s_1k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t * buckets,
    int64_t threads
) {
    libsais16x64_partial_sorting_scan_right_to_left_32s_1k_omp(T, SA, n, buckets, threads, NULL);
}

int64_t probe_libsais16x64_partial_sorting_gather_lms_suffixes_32s_4k(
    int64_t * SA,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_partial_sorting_gather_lms_suffixes_32s_4k(SA, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_partial_sorting_gather_lms_suffixes_32s_1k(
    int64_t * SA,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_partial_sorting_gather_lms_suffixes_32s_1k(SA, omp_block_start, omp_block_size);
}

void probe_libsais16x64_partial_sorting_gather_lms_suffixes_32s_4k_omp(
    int64_t * SA,
    int64_t n,
    int64_t threads
) {
    libsais16x64_partial_sorting_gather_lms_suffixes_32s_4k_omp(SA, n, threads, NULL);
}

void probe_libsais16x64_partial_sorting_gather_lms_suffixes_32s_1k_omp(
    int64_t * SA,
    int64_t n,
    int64_t threads
) {
    libsais16x64_partial_sorting_gather_lms_suffixes_32s_1k_omp(SA, n, threads, NULL);
}

int64_t probe_libsais16x64_count_and_gather_lms_suffixes_32s_2k(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_count_and_gather_lms_suffixes_32s_2k(T, SA, n, k, buckets, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_count_and_gather_compacted_lms_suffixes_32s_2k(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t omp_block_start,
    int64_t omp_block_size
) {
    return libsais16x64_count_and_gather_compacted_lms_suffixes_32s_2k(T, SA, n, k, buckets, omp_block_start, omp_block_size);
}

int64_t probe_libsais16x64_count_and_gather_lms_suffixes_32s_2k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t local_buckets,
    int64_t threads
) {
    return libsais16x64_count_and_gather_lms_suffixes_32s_2k_omp(T, SA, n, k, buckets, local_buckets, threads, NULL);
}

void probe_libsais16x64_count_and_gather_compacted_lms_suffixes_32s_2k_omp(
    const int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t * buckets,
    int64_t local_buckets,
    int64_t threads
) {
    libsais16x64_count_and_gather_compacted_lms_suffixes_32s_2k_omp(T, SA, n, k, buckets, local_buckets, threads, NULL);
}

void probe_libsais16x64_reconstruct_compacted_lms_suffixes_32s_2k_omp(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t m,
    int64_t fs,
    int64_t f,
    int64_t * buckets,
    int64_t local_buckets,
    int64_t threads
) {
    libsais16x64_reconstruct_compacted_lms_suffixes_32s_2k_omp(T, SA, n, k, m, fs, f, buckets, local_buckets, threads, NULL);
}

void probe_libsais16x64_reconstruct_compacted_lms_suffixes_32s_1k_omp(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t m,
    int64_t fs,
    int64_t f,
    int64_t threads
) {
    libsais16x64_reconstruct_compacted_lms_suffixes_32s_1k_omp(T, SA, n, m, fs, f, threads, NULL);
}

int64_t probe_libsais16x64_main_32s_entry(
    int64_t * T,
    int64_t * SA,
    int64_t n,
    int64_t k,
    int64_t fs,
    int64_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16x64_alloc_thread_state(threads) : NULL;
    int64_t result = (thread_state != NULL || threads == 1)
        ? libsais16x64_main_32s_entry(T, SA, n, k, fs, threads, thread_state)
        : -2;
    libsais16x64_free_thread_state(thread_state);
    return result;
}
