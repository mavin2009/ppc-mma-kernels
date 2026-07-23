// xcheck_cache.cpp — pack-cache policy test at N > capacity (N3 in
// VALIDATION-POWER10.md).
//
// Both cache defects in the REVIEW.md log lived in the same regime:
// more tensors than the cache could hold (round-robin thrash, then
// silent 128-slot admission exhaustion).  This test drives that regime
// on purpose:
//
//   1. GROWTH: admit far more tensors than the initial slot table;
//      every second acquire must be a hit.  Slot count must never be
//      the binding constraint.
//   2. CAPACITY: with a tiny byte cap, refusals must be COUNTED (and
//      therefore loud) — a silently uncached tensor is the defect
//      class this file exists to prevent.
//   3. RELOAD: same address, new bytes -> fresh repack, not a stale
//      hit (the fingerprint path).
//
// The defect this would have caught shipped twice.  A capacity policy
// needs a test at N > capacity, not a design review.

#include "ppc_pack_cache.cpp"

#include <cstdio>

#define NKEYS 700           // > 2x the initial slot table
#define PACK_BYTES 4096

static int fails = 0;
static void expect(int cond, const char * what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
}

int main() {
    static uint8_t backing[NKEYS][64];   // fingerprint reads 64 bytes
    for (int i = 0; i < NKEYS; i++)
        for (int j = 0; j < 64; j++) backing[i][j] = (uint8_t)(i ^ j);

    // ---- 1. growth: every tensor admitted, second pass all hits ----
    for (int i = 0; i < NKEYS; i++) {
        int fresh = 0;
        void * p = ppc_apack_cache_acquire(backing[i], 8, 64, 1, PACK_BYTES, &fresh);
        expect(p != NULL, "admission under byte cap must not be refused");
        expect(fresh == 1, "first acquire must be fresh");
        if (p) ppc_apack_cache_publish(backing[i], 8, 64, 1);
    }
    size_t admitted = 0, refused = 0, resident = 0;
    ppc_apack_cache_stats(&admitted, &refused, &resident);
    expect(admitted == NKEYS, "all keys admitted (slot table grew)");
    expect(refused == 0, "no refusals under byte cap");
    expect(resident == (size_t)NKEYS * PACK_BYTES, "resident bytes accounted");
    for (int i = 0; i < NKEYS; i++) {
        int fresh = 1;
        void * p = ppc_apack_cache_acquire(backing[i], 8, 64, 1, PACK_BYTES, &fresh);
        expect(p != NULL && fresh == 0, "second acquire must hit, not repack");
    }
    printf("growth:   %d keys through initial table of 256, all hits on pass 2\n", NKEYS);

    // ---- 2. capacity: refusals are counted, never silent ----
    ppc_apack_cache_clear();
    g_cap = 16 * PACK_BYTES;             // white-box: tiny cap, 16 packs
    size_t admitted0 = admitted;
    for (int i = 0; i < 64; i++) {
        int fresh = 0;
        void * p = ppc_apack_cache_acquire(backing[i], 8, 64, 2, PACK_BYTES, &fresh);
        if (p) ppc_apack_cache_publish(backing[i], 8, 64, 2);
    }
    ppc_apack_cache_stats(&admitted, &refused, &resident);
    expect(admitted - admitted0 == 16, "exactly cap/bytes admissions under tiny cap");
    expect(refused == 64 - 16, "every refusal counted -- silence is the defect");
    printf("capacity: 16/64 admitted at tiny cap, %zu refusals all counted\n", refused);

    // ---- 3. reload detection: same address, new content ----
    int fresh = 0;
    ppc_apack_cache_clear();
    g_cap = (size_t)2048 * 1024 * 1024;
    ppc_apack_cache_acquire(backing[0], 8, 64, 3, PACK_BYTES, &fresh);
    ppc_apack_cache_publish(backing[0], 8, 64, 3);
    backing[0][7] ^= 0xff;               // model reloaded at same address
    ppc_apack_cache_acquire(backing[0], 8, 64, 3, PACK_BYTES, &fresh);
    expect(fresh == 1, "content change at same address must repack");
    ppc_apack_cache_publish(backing[0], 8, 64, 3);
    printf("reload:   fingerprint forces repack on content change\n");

    if (fails) { printf("CACHE TEST FAILED (%d)\n", fails); return 1; }
    printf("ALL CACHE POLICY TESTS PASSED\n");
    return 0;
}
