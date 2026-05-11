#include "../libsais/src/libsais16.c"

int32_t probe_public_libsais16(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t fs
) {
    return libsais16(T, SA, n, fs, NULL);
}

int32_t probe_public_libsais16_freq(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t fs,
    int32_t * freq
) {
    return libsais16(T, SA, n, fs, freq);
}

int32_t probe_public_libsais16_gsa(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t fs
) {
    return libsais16_gsa(T, SA, n, fs, NULL);
}

int32_t probe_public_libsais16_gsa_freq(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t fs,
    int32_t * freq
) {
    return libsais16_gsa(T, SA, n, fs, freq);
}

int32_t probe_public_libsais16_int(
    int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t fs
) {
    return libsais16_int(T, SA, n, k, fs);
}

int32_t probe_libsais16_main_32s_entry(
    int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t fs,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    int32_t result = (thread_state != NULL || threads == 1)
        ? libsais16_main_32s_entry(T, SA, n, k, fs, threads, thread_state)
        : -2;
    libsais16_free_thread_state(thread_state);
    return result;
}

void probe_libsais16_final_sorting_scan_left_to_right_32s(
    const int32_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_final_sorting_scan_left_to_right_32s(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_final_sorting_scan_right_to_left_32s(
    const int32_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_final_sorting_scan_right_to_left_32s(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_clear_lms_suffixes_omp(
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * bucket_start,
    int32_t * bucket_end,
    int32_t threads
) {
    libsais16_clear_lms_suffixes_omp(SA, n, k, bucket_start, bucket_end, threads);
}

void probe_libsais16_flip_suffix_markers_omp(
    int32_t * SA,
    int32_t l,
    int32_t threads
) {
    libsais16_flip_suffix_markers_omp(SA, l, threads);
}

void probe_libsais16_induce_final_order_32s_6k(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16_induce_final_order_32s_6k(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16_free_thread_state(thread_state);
}

void probe_libsais16_induce_final_order_32s_4k(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16_induce_final_order_32s_4k(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16_free_thread_state(thread_state);
}

void probe_libsais16_induce_final_order_32s_2k(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16_induce_final_order_32s_2k(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16_free_thread_state(thread_state);
}

void probe_libsais16_induce_final_order_32s_1k(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16_induce_final_order_32s_1k(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16_free_thread_state(thread_state);
}

void probe_libsais16_induce_partial_order_32s_6k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t first_lms_suffix,
    int32_t left_suffixes_count,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16_induce_partial_order_32s_6k_omp(T, SA, n, k, buckets, first_lms_suffix, left_suffixes_count, threads, thread_state);
    }
    libsais16_free_thread_state(thread_state);
}

void probe_libsais16_induce_partial_order_32s_4k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16_induce_partial_order_32s_4k_omp(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16_free_thread_state(thread_state);
}

void probe_libsais16_induce_partial_order_32s_2k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16_induce_partial_order_32s_2k_omp(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16_free_thread_state(thread_state);
}

void probe_libsais16_induce_partial_order_32s_1k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16_induce_partial_order_32s_1k_omp(T, SA, n, k, buckets, threads, thread_state);
    }
    libsais16_free_thread_state(thread_state);
}

void probe_libsais16_induce_partial_order_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t flags,
    int32_t * buckets,
    int32_t first_lms_suffix,
    int32_t left_suffixes_count,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    if (thread_state != NULL || threads == 1) {
        libsais16_induce_partial_order_16u_omp(T, SA, n, k, flags, buckets, first_lms_suffix, left_suffixes_count, threads, thread_state);
    }
    libsais16_free_thread_state(thread_state);
}

int32_t probe_libsais16_induce_final_order_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t flags,
    int32_t r,
    int32_t * I,
    int32_t * buckets,
    int32_t threads
) {
    LIBSAIS_THREAD_STATE * thread_state = threads > 1 ? libsais16_alloc_thread_state(threads) : NULL;
    int32_t result = (thread_state != NULL || threads == 1)
        ? libsais16_induce_final_order_16u_omp(T, SA, n, k, flags, r, I, buckets, threads, thread_state)
        : -2;
    libsais16_free_thread_state(thread_state);
    return result;
}

int32_t probe_public_libsais16_bwt(
    const uint16_t * T,
    uint16_t * U,
    int32_t * A,
    int32_t n,
    int32_t fs
) {
    return libsais16_bwt(T, U, A, n, fs, NULL);
}

int32_t probe_public_libsais16_bwt_freq(
    const uint16_t * T,
    uint16_t * U,
    int32_t * A,
    int32_t n,
    int32_t fs,
    int32_t * freq
) {
    return libsais16_bwt(T, U, A, n, fs, freq);
}

int32_t probe_public_libsais16_bwt_aux(
    const uint16_t * T,
    uint16_t * U,
    int32_t * A,
    int32_t n,
    int32_t fs,
    int32_t r,
    int32_t * I
) {
    return libsais16_bwt_aux(T, U, A, n, fs, NULL, r, I);
}

int32_t probe_public_libsais16_bwt_aux_freq(
    const uint16_t * T,
    uint16_t * U,
    int32_t * A,
    int32_t n,
    int32_t fs,
    int32_t * freq,
    int32_t r,
    int32_t * I
) {
    return libsais16_bwt_aux(T, U, A, n, fs, freq, r, I);
}

int32_t probe_public_libsais16_unbwt(
    const uint16_t * T,
    uint16_t * U,
    int32_t * A,
    int32_t n,
    int32_t i
) {
    return libsais16_unbwt(T, U, A, n, NULL, i);
}

int32_t probe_public_libsais16_unbwt_freq(
    const uint16_t * T,
    uint16_t * U,
    int32_t * A,
    int32_t n,
    const int32_t * freq,
    int32_t i
) {
    return libsais16_unbwt(T, U, A, n, freq, i);
}

int32_t probe_public_libsais16_unbwt_aux(
    const uint16_t * T,
    uint16_t * U,
    int32_t * A,
    int32_t n,
    int32_t r,
    const int32_t * I
) {
    return libsais16_unbwt_aux(T, U, A, n, NULL, r, I);
}

int32_t probe_public_libsais16_unbwt_aux_freq(
    const uint16_t * T,
    uint16_t * U,
    int32_t * A,
    int32_t n,
    const int32_t * freq,
    int32_t r,
    const int32_t * I
) {
    return libsais16_unbwt_aux(T, U, A, n, freq, r, I);
}

int32_t probe_public_libsais16_plcp(
    const uint16_t * T,
    const int32_t * SA,
    int32_t * PLCP,
    int32_t n
) {
    return libsais16_plcp(T, SA, PLCP, n);
}

int32_t probe_public_libsais16_plcp_gsa(
    const uint16_t * T,
    const int32_t * SA,
    int32_t * PLCP,
    int32_t n
) {
    return libsais16_plcp_gsa(T, SA, PLCP, n);
}

int32_t probe_public_libsais16_lcp(
    const int32_t * PLCP,
    const int32_t * SA,
    int32_t * LCP,
    int32_t n
) {
    return libsais16_lcp(PLCP, SA, LCP, n);
}

void probe_libsais16_gather_lms_suffixes_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_gather_lms_suffixes_16u(T, SA, n, m, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_count_and_gather_lms_suffixes_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t * buckets,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_count_and_gather_lms_suffixes_16u(T, SA, n, buckets, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_initialize_buckets_start_and_end_16u(
    int32_t * buckets,
    int32_t * freq
) {
    return libsais16_initialize_buckets_start_and_end_16u(buckets, freq);
}

int32_t probe_libsais16_initialize_buckets_for_lms_suffixes_radix_sort_16u(
    const uint16_t * T,
    int32_t * buckets,
    int32_t first_lms_suffix
) {
    return libsais16_initialize_buckets_for_lms_suffixes_radix_sort_16u(T, buckets, first_lms_suffix);
}

void probe_libsais16_radix_sort_lms_suffixes_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_radix_sort_lms_suffixes_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_initialize_buckets_for_partial_sorting_16u(
    const uint16_t * T,
    int32_t * buckets,
    int32_t first_lms_suffix,
    int32_t left_suffixes_count
) {
    libsais16_initialize_buckets_for_partial_sorting_16u(T, buckets, first_lms_suffix, left_suffixes_count);
}

int32_t probe_libsais16_partial_sorting_scan_left_to_right_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t * buckets,
    int32_t d,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_partial_sorting_scan_left_to_right_16u(T, SA, buckets, d, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_partial_sorting_scan_right_to_left_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t * buckets,
    int32_t d,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_partial_sorting_scan_right_to_left_16u(T, SA, buckets, d, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_partial_gsa_scan_right_to_left_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t * buckets,
    int32_t d,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_partial_gsa_scan_right_to_left_16u(T, SA, buckets, d, omp_block_start, omp_block_size);
}

void probe_libsais16_partial_sorting_shift_markers_16u_omp(
    int32_t * SA,
    int32_t n,
    const int32_t * buckets,
    int32_t threads
) {
    libsais16_partial_sorting_shift_markers_16u_omp(SA, n, buckets, threads);
}

void probe_libsais16_final_sorting_scan_left_to_right_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_final_sorting_scan_left_to_right_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_final_sorting_scan_right_to_left_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_final_sorting_scan_right_to_left_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_final_gsa_scan_right_to_left_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_final_gsa_scan_right_to_left_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_final_bwt_scan_left_to_right_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_final_bwt_scan_left_to_right_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_final_bwt_scan_right_to_left_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_final_bwt_scan_right_to_left_16u(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_final_bwt_aux_scan_left_to_right_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t rm,
    int32_t * I,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_final_bwt_aux_scan_left_to_right_16u(T, SA, rm, I, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_final_bwt_aux_scan_right_to_left_16u(
    const uint16_t * T,
    int32_t * SA,
    int32_t rm,
    int32_t * I,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_final_bwt_aux_scan_right_to_left_16u(T, SA, rm, I, induction_bucket, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_renumber_lms_suffixes_16u(
    int32_t * SA,
    int32_t m,
    int32_t name,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_renumber_lms_suffixes_16u(SA, m, name, omp_block_start, omp_block_size);
}

void probe_libsais16_place_lms_suffixes_interval_16u(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t flags,
    int32_t * buckets
) {
    libsais16_place_lms_suffixes_interval_16u(SA, n, m, flags, buckets);
}

void probe_libsais16_bwt_copy_16u(
    uint16_t * U,
    int32_t * A,
    int32_t n
) {
    libsais16_bwt_copy_16u(U, A, n);
}

void probe_libsais16_gather_lms_suffixes_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t threads
) {
    libsais16_gather_lms_suffixes_16u_omp(T, SA, n, threads, NULL);
}

int32_t probe_libsais16_count_and_gather_lms_suffixes_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t * buckets,
    int32_t threads
) {
    return libsais16_count_and_gather_lms_suffixes_16u_omp(T, SA, n, buckets, threads, NULL);
}

void probe_libsais16_radix_sort_lms_suffixes_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t flags,
    int32_t * buckets,
    int32_t threads
) {
    libsais16_radix_sort_lms_suffixes_16u_omp(T, SA, n, m, flags, buckets, threads, NULL);
}

int32_t probe_libsais16_partial_sorting_scan_left_to_right_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t left_suffixes_count,
    int32_t d,
    int32_t threads
) {
    return libsais16_partial_sorting_scan_left_to_right_16u_omp(T, SA, n, k, buckets, left_suffixes_count, d, threads, NULL);
}

void probe_libsais16_partial_sorting_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t first_lms_suffix,
    int32_t left_suffixes_count,
    int32_t d,
    int32_t threads
) {
    libsais16_partial_sorting_scan_right_to_left_16u_omp(T, SA, n, k, buckets, first_lms_suffix, left_suffixes_count, d, threads, NULL);
}

void probe_libsais16_partial_gsa_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t first_lms_suffix,
    int32_t left_suffixes_count,
    int32_t d,
    int32_t threads
) {
    libsais16_partial_gsa_scan_right_to_left_16u_omp(T, SA, n, k, buckets, first_lms_suffix, left_suffixes_count, d, threads, NULL);
}

int32_t probe_libsais16_renumber_lms_suffixes_16u_omp(
    int32_t * SA,
    int32_t m,
    int32_t threads
) {
    return libsais16_renumber_lms_suffixes_16u_omp(SA, m, threads, NULL);
}

void probe_libsais16_final_bwt_scan_left_to_right_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_final_bwt_scan_left_to_right_16u_omp(T, SA, n, k, induction_bucket, threads, NULL);
}

void probe_libsais16_final_bwt_aux_scan_left_to_right_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t rm,
    int32_t * I,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_final_bwt_aux_scan_left_to_right_16u_omp(T, SA, n, k, rm, I, induction_bucket, threads, NULL);
}

void probe_libsais16_final_sorting_scan_left_to_right_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_final_sorting_scan_left_to_right_16u_omp(T, SA, n, k, induction_bucket, threads, NULL);
}

int32_t probe_libsais16_final_bwt_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * induction_bucket,
    int32_t threads
) {
    return libsais16_final_bwt_scan_right_to_left_16u_omp(T, SA, n, k, induction_bucket, threads, NULL);
}

void probe_libsais16_final_bwt_aux_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t rm,
    int32_t * I,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_final_bwt_aux_scan_right_to_left_16u_omp(T, SA, n, k, rm, I, induction_bucket, threads, NULL);
}

void probe_libsais16_final_sorting_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t omp_block_start,
    int32_t omp_block_size,
    int32_t k,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_final_sorting_scan_right_to_left_16u_omp(T, SA, omp_block_start, omp_block_size, k, induction_bucket, threads, NULL);
}

void probe_libsais16_final_gsa_scan_right_to_left_16u_omp(
    const uint16_t * T,
    int32_t * SA,
    int32_t omp_block_start,
    int32_t omp_block_size,
    int32_t k,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_final_gsa_scan_right_to_left_16u_omp(T, SA, omp_block_start, omp_block_size, k, induction_bucket, threads, NULL);
}

void probe_libsais16_bwt_copy_16u_omp(
    uint16_t * U,
    int32_t * A,
    int32_t n,
    int32_t threads
) {
    libsais16_bwt_copy_16u_omp(U, A, n, threads);
}

int32_t probe_libsais16_gather_marked_lms_suffixes(
    int32_t * SA,
    int32_t m,
    int32_t l,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return (int32_t)libsais16_gather_marked_lms_suffixes(SA, m, l, omp_block_start, omp_block_size);
}

void probe_libsais16_gather_marked_lms_suffixes_omp(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t fs,
    int32_t threads
) {
    libsais16_gather_marked_lms_suffixes_omp(SA, n, m, fs, threads, NULL);
}

int32_t probe_libsais16_renumber_and_gather_lms_suffixes_omp(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t fs,
    int32_t threads
) {
    return libsais16_renumber_and_gather_lms_suffixes_omp(SA, n, m, fs, threads, NULL);
}

void probe_libsais16_reconstruct_lms_suffixes(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_reconstruct_lms_suffixes(SA, n, m, omp_block_start, omp_block_size);
}

void probe_libsais16_reconstruct_lms_suffixes_omp(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t threads
) {
    libsais16_reconstruct_lms_suffixes_omp(SA, n, m, threads);
}

int32_t probe_libsais16_renumber_distinct_lms_suffixes_32s_4k(
    int32_t * SA,
    int32_t m,
    int32_t name,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_renumber_distinct_lms_suffixes_32s_4k(SA, m, name, omp_block_start, omp_block_size);
}

void probe_libsais16_mark_distinct_lms_suffixes_32s(
    int32_t * SA,
    int32_t m,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_mark_distinct_lms_suffixes_32s(SA, m, omp_block_start, omp_block_size);
}

void probe_libsais16_clamp_lms_suffixes_length_32s(
    int32_t * SA,
    int32_t m,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_clamp_lms_suffixes_length_32s(SA, m, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_renumber_distinct_lms_suffixes_32s_4k_omp(
    int32_t * SA,
    int32_t m,
    int32_t threads
) {
    return libsais16_renumber_distinct_lms_suffixes_32s_4k_omp(SA, m, threads, NULL);
}

void probe_libsais16_mark_distinct_lms_suffixes_32s_omp(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t threads
) {
    libsais16_mark_distinct_lms_suffixes_32s_omp(SA, n, m, threads);
}

void probe_libsais16_clamp_lms_suffixes_length_32s_omp(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t threads
) {
    libsais16_clamp_lms_suffixes_length_32s_omp(SA, n, m, threads);
}

int32_t probe_libsais16_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t threads
) {
    return libsais16_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(SA, n, m, threads, NULL);
}

int32_t probe_libsais16_renumber_unique_and_nonunique_lms_suffixes_32s(
    int32_t * T,
    int32_t * SA,
    int32_t m,
    int32_t f,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_renumber_unique_and_nonunique_lms_suffixes_32s(T, SA, m, f, omp_block_start, omp_block_size);
}

void probe_libsais16_compact_unique_and_nonunique_lms_suffixes_32s(
    int32_t * SA,
    int32_t m,
    int32_t * pl,
    int32_t * pr,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    fast_sint_t l = *pl;
    fast_sint_t r = *pr;
    libsais16_compact_unique_and_nonunique_lms_suffixes_32s(SA, m, &l, &r, omp_block_start, omp_block_size);
    *pl = (int32_t)l;
    *pr = (int32_t)r;
}

int32_t probe_libsais16_renumber_unique_and_nonunique_lms_suffixes_32s_omp(
    int32_t * T,
    int32_t * SA,
    int32_t m,
    int32_t threads
) {
    return libsais16_renumber_unique_and_nonunique_lms_suffixes_32s_omp(T, SA, m, threads, NULL);
}

void probe_libsais16_compact_unique_and_nonunique_lms_suffixes_32s_omp(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t fs,
    int32_t f,
    int32_t threads
) {
    libsais16_compact_unique_and_nonunique_lms_suffixes_32s_omp(SA, n, m, fs, f, threads, NULL);
}

int32_t probe_libsais16_compact_lms_suffixes_32s_omp(
    int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t fs,
    int32_t threads
) {
    return libsais16_compact_lms_suffixes_32s_omp(T, SA, n, m, fs, threads, NULL);
}

void probe_libsais16_merge_unique_lms_suffixes_32s(
    int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t l,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_merge_unique_lms_suffixes_32s(T, SA, n, m, l, omp_block_start, omp_block_size);
}

void probe_libsais16_merge_nonunique_lms_suffixes_32s(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t l,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_merge_nonunique_lms_suffixes_32s(SA, n, m, l, omp_block_start, omp_block_size);
}

void probe_libsais16_merge_unique_lms_suffixes_32s_omp(
    int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t threads
) {
    libsais16_merge_unique_lms_suffixes_32s_omp(T, SA, n, m, threads, NULL);
}

void probe_libsais16_merge_nonunique_lms_suffixes_32s_omp(
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t f,
    int32_t threads
) {
    libsais16_merge_nonunique_lms_suffixes_32s_omp(SA, n, m, f, threads, NULL);
}

void probe_libsais16_merge_compacted_lms_suffixes_32s_omp(
    int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t f,
    int32_t threads
) {
    libsais16_merge_compacted_lms_suffixes_32s_omp(T, SA, n, m, f, threads, NULL);
}

void probe_libsais16_radix_sort_lms_suffixes_32s_6k(
    const int32_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_radix_sort_lms_suffixes_32s_6k(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_radix_sort_lms_suffixes_32s_2k(
    const int32_t * T,
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_radix_sort_lms_suffixes_32s_2k(T, SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_radix_sort_lms_suffixes_32s_6k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_radix_sort_lms_suffixes_32s_6k_omp(T, SA, n, m, induction_bucket, threads, NULL);
}

void probe_libsais16_radix_sort_lms_suffixes_32s_2k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_radix_sort_lms_suffixes_32s_2k_omp(T, SA, n, m, induction_bucket, threads, NULL);
}

int32_t probe_libsais16_radix_sort_lms_suffixes_32s_1k(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t * buckets
) {
    return libsais16_radix_sort_lms_suffixes_32s_1k(T, SA, n, buckets);
}

void probe_libsais16_radix_sort_set_markers_32s_6k(
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_radix_sort_set_markers_32s_6k(SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_radix_sort_set_markers_32s_4k(
    int32_t * SA,
    int32_t * induction_bucket,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_radix_sort_set_markers_32s_4k(SA, induction_bucket, omp_block_start, omp_block_size);
}

void probe_libsais16_radix_sort_set_markers_32s_6k_omp(
    int32_t * SA,
    int32_t k,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_radix_sort_set_markers_32s_6k_omp(SA, k, induction_bucket, threads);
}

void probe_libsais16_radix_sort_set_markers_32s_4k_omp(
    int32_t * SA,
    int32_t k,
    int32_t * induction_bucket,
    int32_t threads
) {
    libsais16_radix_sort_set_markers_32s_4k_omp(SA, k, induction_bucket, threads);
}

void probe_libsais16_place_lms_suffixes_histogram_32s_6k(
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t m,
    const int32_t * buckets
) {
    libsais16_place_lms_suffixes_histogram_32s_6k(SA, n, k, m, buckets);
}

void probe_libsais16_place_lms_suffixes_histogram_32s_4k(
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t m,
    const int32_t * buckets
) {
    libsais16_place_lms_suffixes_histogram_32s_4k(SA, n, k, m, buckets);
}

void probe_libsais16_place_lms_suffixes_histogram_32s_2k(
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t m,
    const int32_t * buckets
) {
    libsais16_place_lms_suffixes_histogram_32s_2k(SA, n, k, m, buckets);
}

int32_t probe_libsais16_gather_lms_suffixes_32s(
    const int32_t * T,
    int32_t * SA,
    int32_t n
) {
    return libsais16_gather_lms_suffixes_32s(T, SA, n);
}

int32_t probe_libsais16_gather_compacted_lms_suffixes_32s(
    const int32_t * T,
    int32_t * SA,
    int32_t n
) {
    return libsais16_gather_compacted_lms_suffixes_32s(T, SA, n);
}

void probe_libsais16_count_lms_suffixes_32s_2k(
    const int32_t * T,
    int32_t n,
    int32_t k,
    int32_t * buckets
) {
    libsais16_count_lms_suffixes_32s_2k(T, n, k, buckets);
}

int32_t probe_libsais16_count_and_gather_lms_suffixes_32s_4k(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_count_and_gather_lms_suffixes_32s_4k(T, SA, n, k, buckets, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_count_and_gather_lms_suffixes_32s_4k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t local_buckets,
    int32_t threads
) {
    return libsais16_count_and_gather_lms_suffixes_32s_4k_omp(T, SA, n, k, buckets, local_buckets, threads, NULL);
}

void probe_libsais16_count_suffixes_32s(
    const int32_t * T,
    int32_t n,
    int32_t k,
    int32_t * buckets
) {
    libsais16_count_suffixes_32s(T, n, k, buckets);
}

void probe_libsais16_initialize_buckets_start_and_end_32s_6k(
    int32_t k,
    int32_t * buckets
) {
    libsais16_initialize_buckets_start_and_end_32s_6k(k, buckets);
}

void probe_libsais16_initialize_buckets_start_and_end_32s_4k(
    int32_t k,
    int32_t * buckets
) {
    libsais16_initialize_buckets_start_and_end_32s_4k(k, buckets);
}

void probe_libsais16_initialize_buckets_end_32s_2k(
    int32_t k,
    int32_t * buckets
) {
    libsais16_initialize_buckets_end_32s_2k(k, buckets);
}

void probe_libsais16_initialize_buckets_start_and_end_32s_2k(
    int32_t k,
    int32_t * buckets
) {
    libsais16_initialize_buckets_start_and_end_32s_2k(k, buckets);
}

void probe_libsais16_initialize_buckets_start_32s_1k(
    int32_t k,
    int32_t * buckets
) {
    libsais16_initialize_buckets_start_32s_1k(k, buckets);
}

void probe_libsais16_initialize_buckets_end_32s_1k(
    int32_t k,
    int32_t * buckets
) {
    libsais16_initialize_buckets_end_32s_1k(k, buckets);
}

void probe_libsais16_initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(
    const int32_t * T,
    int32_t k,
    int32_t * buckets,
    int32_t first_lms_suffix
) {
    libsais16_initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(T, k, buckets, first_lms_suffix);
}

int32_t probe_libsais16_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(
    const int32_t * T,
    int32_t k,
    int32_t * buckets,
    int32_t first_lms_suffix
) {
    return libsais16_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(T, k, buckets, first_lms_suffix);
}

void probe_libsais16_initialize_buckets_for_radix_and_partial_sorting_32s_4k(
    const int32_t * T,
    int32_t k,
    int32_t * buckets,
    int32_t first_lms_suffix
) {
    libsais16_initialize_buckets_for_radix_and_partial_sorting_32s_4k(T, k, buckets, first_lms_suffix);
}

void probe_libsais16_place_lms_suffixes_interval_32s_4k(
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t m,
    const int32_t * buckets
) {
    libsais16_place_lms_suffixes_interval_32s_4k(SA, n, k, m, buckets);
}

void probe_libsais16_place_lms_suffixes_interval_32s_2k(
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t m,
    const int32_t * buckets
) {
    libsais16_place_lms_suffixes_interval_32s_2k(SA, n, k, m, buckets);
}

void probe_libsais16_place_lms_suffixes_interval_32s_1k(
    const int32_t * T,
    int32_t * SA,
    int32_t k,
    int32_t m,
    int32_t * buckets
) {
    libsais16_place_lms_suffixes_interval_32s_1k(T, SA, k, m, buckets);
}

int32_t probe_libsais16_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(
    int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t threads
) {
    return libsais16_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(T, SA, n, m, threads);
}

void probe_libsais16_partial_sorting_shift_markers_32s_6k_omp(
    int32_t * SA,
    int32_t k,
    const int32_t * buckets,
    int32_t threads
) {
    libsais16_partial_sorting_shift_markers_32s_6k_omp(SA, k, buckets, threads);
}

void probe_libsais16_partial_sorting_shift_markers_32s_4k(
    int32_t * SA,
    int32_t n
) {
    libsais16_partial_sorting_shift_markers_32s_4k(SA, n);
}

void probe_libsais16_partial_sorting_shift_buckets_32s_6k(
    int32_t k,
    int32_t * buckets
) {
    libsais16_partial_sorting_shift_buckets_32s_6k(k, buckets);
}

int32_t probe_libsais16_partial_sorting_scan_left_to_right_32s_6k(
    const int32_t * T,
    int32_t * SA,
    int32_t * buckets,
    int32_t d,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_partial_sorting_scan_left_to_right_32s_6k(T, SA, buckets, d, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_partial_sorting_scan_left_to_right_32s_4k(
    const int32_t * T,
    int32_t * SA,
    int32_t k,
    int32_t * buckets,
    int32_t d,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_partial_sorting_scan_left_to_right_32s_4k(T, SA, k, buckets, d, omp_block_start, omp_block_size);
}

void probe_libsais16_partial_sorting_scan_left_to_right_32s_1k(
    const int32_t * T,
    int32_t * SA,
    int32_t * buckets,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_partial_sorting_scan_left_to_right_32s_1k(T, SA, buckets, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_partial_sorting_scan_left_to_right_32s_6k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t * buckets,
    int32_t left_suffixes_count,
    int32_t d,
    int32_t threads
) {
    return libsais16_partial_sorting_scan_left_to_right_32s_6k_omp(T, SA, n, buckets, left_suffixes_count, d, threads, NULL);
}

int32_t probe_libsais16_partial_sorting_scan_left_to_right_32s_4k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t d,
    int32_t threads
) {
    return libsais16_partial_sorting_scan_left_to_right_32s_4k_omp(T, SA, n, k, buckets, d, threads, NULL);
}

void probe_libsais16_partial_sorting_scan_left_to_right_32s_1k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t * buckets,
    int32_t threads
) {
    libsais16_partial_sorting_scan_left_to_right_32s_1k_omp(T, SA, n, buckets, threads, NULL);
}

int32_t probe_libsais16_partial_sorting_scan_right_to_left_32s_6k(
    const int32_t * T,
    int32_t * SA,
    int32_t * buckets,
    int32_t d,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_partial_sorting_scan_right_to_left_32s_6k(T, SA, buckets, d, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_partial_sorting_scan_right_to_left_32s_4k(
    const int32_t * T,
    int32_t * SA,
    int32_t k,
    int32_t * buckets,
    int32_t d,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_partial_sorting_scan_right_to_left_32s_4k(T, SA, k, buckets, d, omp_block_start, omp_block_size);
}

void probe_libsais16_partial_sorting_scan_right_to_left_32s_1k(
    const int32_t * T,
    int32_t * SA,
    int32_t * buckets,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    libsais16_partial_sorting_scan_right_to_left_32s_1k(T, SA, buckets, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_partial_sorting_scan_right_to_left_32s_6k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t * buckets,
    int32_t first_lms_suffix,
    int32_t left_suffixes_count,
    int32_t d,
    int32_t threads
) {
    return libsais16_partial_sorting_scan_right_to_left_32s_6k_omp(T, SA, n, buckets, first_lms_suffix, left_suffixes_count, d, threads, NULL);
}

int32_t probe_libsais16_partial_sorting_scan_right_to_left_32s_4k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t d,
    int32_t threads
) {
    return libsais16_partial_sorting_scan_right_to_left_32s_4k_omp(T, SA, n, k, buckets, d, threads, NULL);
}

void probe_libsais16_partial_sorting_scan_right_to_left_32s_1k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t * buckets,
    int32_t threads
) {
    libsais16_partial_sorting_scan_right_to_left_32s_1k_omp(T, SA, n, buckets, threads, NULL);
}

int32_t probe_libsais16_partial_sorting_gather_lms_suffixes_32s_4k(
    int32_t * SA,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_partial_sorting_gather_lms_suffixes_32s_4k(SA, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_partial_sorting_gather_lms_suffixes_32s_1k(
    int32_t * SA,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_partial_sorting_gather_lms_suffixes_32s_1k(SA, omp_block_start, omp_block_size);
}

void probe_libsais16_partial_sorting_gather_lms_suffixes_32s_4k_omp(
    int32_t * SA,
    int32_t n,
    int32_t threads
) {
    libsais16_partial_sorting_gather_lms_suffixes_32s_4k_omp(SA, n, threads, NULL);
}

void probe_libsais16_partial_sorting_gather_lms_suffixes_32s_1k_omp(
    int32_t * SA,
    int32_t n,
    int32_t threads
) {
    libsais16_partial_sorting_gather_lms_suffixes_32s_1k_omp(SA, n, threads, NULL);
}

int32_t probe_libsais16_count_and_gather_lms_suffixes_32s_2k(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_count_and_gather_lms_suffixes_32s_2k(T, SA, n, k, buckets, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_count_and_gather_compacted_lms_suffixes_32s_2k(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t omp_block_start,
    int32_t omp_block_size
) {
    return libsais16_count_and_gather_compacted_lms_suffixes_32s_2k(T, SA, n, k, buckets, omp_block_start, omp_block_size);
}

int32_t probe_libsais16_count_and_gather_lms_suffixes_32s_2k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t local_buckets,
    int32_t threads
) {
    return libsais16_count_and_gather_lms_suffixes_32s_2k_omp(T, SA, n, k, buckets, local_buckets, threads, NULL);
}

void probe_libsais16_count_and_gather_compacted_lms_suffixes_32s_2k_omp(
    const int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t * buckets,
    int32_t local_buckets,
    int32_t threads
) {
    libsais16_count_and_gather_compacted_lms_suffixes_32s_2k_omp(T, SA, n, k, buckets, local_buckets, threads, NULL);
}

void probe_libsais16_reconstruct_compacted_lms_suffixes_32s_2k_omp(
    int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t k,
    int32_t m,
    int32_t fs,
    int32_t f,
    int32_t * buckets,
    int32_t local_buckets,
    int32_t threads
) {
    libsais16_reconstruct_compacted_lms_suffixes_32s_2k_omp(T, SA, n, k, m, fs, f, buckets, local_buckets, threads, NULL);
}

void probe_libsais16_reconstruct_compacted_lms_suffixes_32s_1k_omp(
    int32_t * T,
    int32_t * SA,
    int32_t n,
    int32_t m,
    int32_t fs,
    int32_t f,
    int32_t threads
) {
    libsais16_reconstruct_compacted_lms_suffixes_32s_1k_omp(T, SA, n, m, fs, f, threads, NULL);
}
