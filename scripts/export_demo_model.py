"""Measure how small the ngram model can get if we ship only what a live keyboard demo
needs: the top-K words per (context, prefix), ranked by count, not raw counts themselves.
predict_word()/topk_by_letter() in infer_ngram.py never score arbitrary candidates -- they
only rank what's already in the count table -- so once that ranking is resolved offline,
the counts themselves are dead weight; only the word ids in rank order matter.

--max-prefix-len > 1 keys the table by every prefix length 1..max_prefix_len instead of just
the first letter, so a live demo can narrow suggestions as more of the word is typed
("st" -> "star", not just "s" -> whatever wins on 's' alone).

--top-k > 1 keeps that many ranked candidates per (context, prefix) instead of just the
argmax, so a live demo can show e.g. 3 suggestions instead of 1.

--min-count applies the same "keep intact if it would empty a context" exception
train_ngram.py's own --min-count used at *training* time: a context is only filtered down to
its >=min_count candidates if at least one candidate actually clears that bar; if every
candidate is rare, the context keeps all of them, unfiltered. Per report.md, 98% of order-4 /
94% of order-3 contexts in ngram_4_a.bin survive training's own pruning ONLY because of this
exact exception -- an earlier version of this script re-applied the floor with no exception
and silently emptied nearly all of those contexts (real predict_word never filters by count
at inference, so it still returns an answer there). With the exception, the #1 (highest-count)
candidate is *always* preserved for every context regardless of --min-count, since the max
count in a context always clears that context's own effective floor -- so raising --min-count
only ever trims lower-ranked (2nd/3rd, when --top-k > 1) candidates in otherwise-healthy
contexts, never the top prediction.

Usage:
    python3 export_demo_model.py --model ../weights/ngram_4_a.bin --max-prefix-len 1 --top-k 3 --min-count 10
"""
import argparse
import gzip
import heapq
import pickle
import struct


def load_model(path):
    with open(path, "rb") as f:
        m = pickle.load(f)
    return m["n"], m["counts"], m["vocab"], m["id_to_tok"]


def build_topk_table(d_ctx, id_to_tok, min_count, max_prefix_len, top_k, strict=False):
    """For one order's count table {ctx: {word_id: count}}, return
    {ctx: {prefix: (word_id, ...)}} -- up to top_k word ids, ranked by count descending,
    for every prefix length 1..min(max_prefix_len, len(word)).

    Keeps a size-bounded min-heap per prefix (top_k entries, not all candidates) --
    at min_count=1 an unpruned order-4 table has 43M+ surviving (ctx, word) pairs, and
    accumulating every one before sorting+slicing blew past 27GB RSS and had to be killed.
    O(top_k) per prefix instead of O(candidates) is the actual fix; this is not a
    Python-vs-C problem, the count tables were never the thing exploding.

    strict=False (default): "keep intact" exception, same as train_ngram.py's
    _compact(hard=False) -- if EVERY candidate in a context is below min_count, the floor
    does not apply there (keeps everyone instead of dropping the context), since real
    predict_word never filters by count at inference and this guarantees the #1 answer
    always matches it regardless of min_count.
    strict=True: unconditional floor, same as _compact(hard=True) -- a context where nothing
    clears min_count is just dropped (falls back to the next-lower order at inference,
    same as train_ngram.py's own --prune-every-lines checkpoints already do). A deliberate
    choice to treat sub-threshold examples as too sparse to matter, not a safety net."""
    out = {}
    for ctx, cand in d_ctx.items():
        eff_min = min_count if (strict or max(cand.values()) >= min_count) else 1
        by_prefix = {}
        for wid, c in cand.items():
            if c < eff_min:
                continue
            tok = id_to_tok[wid]
            if not tok:
                continue
            for L in range(1, min(max_prefix_len, len(tok)) + 1):
                heap = by_prefix.setdefault(tok[:L], [])
                if len(heap) < top_k:
                    heapq.heappush(heap, (c, wid))
                elif c > heap[0][0]:
                    heapq.heapreplace(heap, (c, wid))
        if by_prefix:
            out[ctx] = {
                prefix: tuple(wid for _c, wid in sorted(heap, reverse=True))
                for prefix, heap in by_prefix.items()
            }
    return out


def build_streaming(d_ctx, id_to_tok, min_count, max_prefix_len, top_k, want_trie, strict=False):
    """Same result as build_topk_table + pack_one + build_trie combined, but in one pass that
    never holds a full {ctx: {prefix: wids}} table or a full list of row tuples in memory:

    - pops each context out of d_ctx as it's processed, so the raw input shrinks instead of
      coexisting with everything being built from it
    - packs each row's bytes straight into a bytearray as soon as it's ranked, instead of
      collecting row tuples into a list first (order-4 at min-count=1 has ~10^8-scale
      (ctx,prefix) pairs -- that many live Python tuples is 10s of GB just in object
      overhead; the same data as packed bytes is a few GB)
    - inserts into the trie in the same per-context pass instead of building it from a
      separately-held table afterward

    Trade: flat_buf is in raw dict-iteration order, not globally sorted by (ctx, prefix) --
    fine for measuring size (this function's actual job), not sorted enough for real binary
    search yet. That's a separate concern for whenever the JS reader gets built, not this
    measurement pass."""
    flat_buf = bytearray()
    trie = {} if want_trie else None
    for ctx in list(d_ctx.keys()):
        cand = d_ctx.pop(ctx)
        # keep-intact exception (or not, if strict) -- see build_topk_table's docstring
        eff_min = min_count if (strict or max(cand.values()) >= min_count) else 1
        by_prefix = {}
        for wid, c in cand.items():
            if c < eff_min:
                continue
            tok = id_to_tok[wid]
            if not tok:
                continue
            for L in range(1, min(max_prefix_len, len(tok)) + 1):
                heap = by_prefix.setdefault(tok[:L], [])
                if len(heap) < top_k:
                    heapq.heappush(heap, (c, wid))
                elif c > heap[0][0]:
                    heapq.heapreplace(heap, (c, wid))
        if not by_prefix:
            continue

        if want_trie:
            ctx_node = trie
            for cid in ctx:
                ctx_node = ctx_node.setdefault(cid, {})

        for prefix, heap in by_prefix.items():
            wids = tuple(wid for _c, wid in sorted(heap, reverse=True))

            for cid in ctx:
                flat_buf += struct.pack("<I", cid)
            pbytes = prefix.encode("utf-8")
            flat_buf += struct.pack("<B", len(pbytes))
            flat_buf += pbytes
            flat_buf += struct.pack("<B", len(wids))
            for wid in wids:
                flat_buf += struct.pack("<I", wid)

            if want_trie:
                node = ctx_node
                for ch in prefix:
                    node = node.setdefault(ch, {})
                node[VALUE_KEY] = wids

    return bytes(flat_buf), trie


def _iter_flat_rows(buf, ctx_len):
    """Inverse of the flat packing in build_streaming/pack_one -- decodes (ctx, prefix, wids)
    rows back out. Only used to compare build_streaming's output against build_topk_table's
    on real data (they emit rows in different order, so compare as decoded sets, not bytes)."""
    offset = 0
    while offset < len(buf):
        ctx = struct.unpack_from(f"<{ctx_len}I", buf, offset)
        offset += 4 * ctx_len
        plen = buf[offset]
        offset += 1
        prefix = buf[offset:offset + plen].decode("utf-8")
        offset += plen
        n_wids = buf[offset]
        offset += 1
        wids = struct.unpack_from(f"<{n_wids}I", buf, offset)
        offset += 4 * n_wids
        yield (ctx, prefix, wids)


def pack_one(table):
    """Binary for a single order's table: sorted (ctx ids..., prefix) -> ranked word_ids.
    ctx ids and word_ids are uint32; prefix is length-prefixed UTF-8 bytes (1 byte length +
    bytes, variable-length since padding to a fixed width would waste space for no reason
    here); word_id list is 1 byte count-of-ids + that many uint32s (rank order = array order,
    no separate score needed -- nothing downstream ever compares two entries' counts, only
    iterates rank order)."""
    rows = []
    for ctx, by_prefix in table.items():
        for prefix, wids in by_prefix.items():
            rows.append((ctx, prefix, wids))
    rows.sort()
    buf = bytearray()
    for ctx, prefix, wids in rows:
        for cid in ctx:
            buf += struct.pack("<I", cid)
        pbytes = prefix.encode("utf-8")
        buf += struct.pack("<B", len(pbytes))
        buf += pbytes
        buf += struct.pack("<B", len(wids))
        for wid in wids:
            buf += struct.pack("<I", wid)
    return bytes(buf), len(rows)


VALUE_KEY = ...  # Ellipsis: sentinel dict key, can't collide with an int ctx id or a 1-char str


def build_trie(table):
    """Same {ctx: {prefix: word_ids}} table, reshaped into one nested dict: edges are context
    ids first (one edge per token, so contexts sharing a prefix of tokens share nodes), then
    prefix characters (one edge per char, so "s"/"st"/"sto" share the "s"->"t" chain instead of
    each repeating the full context). A node's VALUE_KEY holds that node's word_ids, if any
    prefix/context combo ends exactly there."""
    root = {}
    for ctx, by_prefix in table.items():
        ctx_node = root
        for cid in ctx:
            ctx_node = ctx_node.setdefault(cid, {})
        for prefix, wids in by_prefix.items():
            node = ctx_node
            for ch in prefix:
                node = node.setdefault(ch, {})
            node[VALUE_KEY] = wids
    return root


def encode_trie(node):
    """Recursive length-prefixed encoding: [has_value(1B)][word_ids if any][n_children(4B)]
    then per child [tag(1B): 0=int ctx-id edge, 1=char edge][edge key][child_len(4B)][child bytes].
    Length-prefixed (not absolute offsets) -- simplest thing that lets a reader skip a subtree
    without parsing it, no separate pointer table needed."""
    buf = bytearray()
    wids = node.get(VALUE_KEY)
    if wids is not None:
        buf += struct.pack("<B", 1)
        buf += struct.pack("<B", len(wids))
        for wid in wids:
            buf += struct.pack("<I", wid)
    else:
        buf += struct.pack("<B", 0)

    children = [(k, v) for k, v in node.items() if k is not VALUE_KEY]
    buf += struct.pack("<I", len(children))  # root nodes can have ~99k children (vocab-sized), past uint16
    for key, child in children:
        child_bytes = encode_trie(child)
        if isinstance(key, int):
            buf += struct.pack("<B", 0)
            buf += struct.pack("<I", key)
        else:
            kb = key.encode("utf-8")
            buf += struct.pack("<B", 1)
            buf += struct.pack("<B", len(kb))
            buf += kb
        buf += struct.pack("<I", len(child_bytes))
        buf += child_bytes
    return bytes(buf)


def decode_trie(buf, offset=0):
    """Inverse of encode_trie -- reconstructs the same nested-dict shape build_trie produced.
    Only used by the self-check here; the real reader is whatever the JS port implements."""
    has_value = buf[offset]
    offset += 1
    node = {}
    if has_value:
        n_ids = buf[offset]
        offset += 1
        wids = struct.unpack_from(f"<{n_ids}I", buf, offset)
        offset += 4 * n_ids
        node[VALUE_KEY] = wids

    n_children = struct.unpack_from("<I", buf, offset)[0]
    offset += 4
    for _ in range(n_children):
        tag = buf[offset]
        offset += 1
        if tag == 0:
            key = struct.unpack_from("<I", buf, offset)[0]
            offset += 4
        else:
            klen = buf[offset]
            offset += 1
            key = buf[offset:offset + klen].decode("utf-8")
            offset += klen
        child_len = struct.unpack_from("<I", buf, offset)[0]
        offset += 4
        child_node, _ = decode_trie(buf, offset)
        node[key] = child_node
        offset += child_len
    return node, offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../weights/ngram_4_a.bin")
    ap.add_argument("--min-count", type=int, default=1,
                     help="extra floor on top of whatever training already pruned. Safe at any "
                          "value -- a context where every candidate is below this floor keeps "
                          "all of them (same 'keep intact' exception train_ngram.py used), so "
                          "the #1 prediction always matches predict_word regardless of this "
                          "value. Higher values only trim size by dropping 2nd/3rd-ranked "
                          "candidates in otherwise-healthy contexts.")
    ap.add_argument("--max-prefix-len", type=int, default=1,
                     help="1 = first-letter only (original behavior). >1 also keys the table "
                          "by longer prefixes so suggestions narrow as more letters are typed.")
    ap.add_argument("--top-k", type=int, default=1,
                     help="1 = argmax only (original behavior). >1 keeps that many ranked "
                          "candidates per (context, prefix), e.g. 3 for a 3-suggestion demo.")
    ap.add_argument("--trie", action="store_true",
                     help="also encode as a shared-prefix trie (ctx ids then prefix chars) and "
                          "compare size against the flat sorted-array encoding")
    ap.add_argument("--strict", action="store_true",
                     help="unconditional --min-count floor, no keep-intact exception -- a "
                          "context where nothing clears the floor is dropped entirely instead "
                          "of kept unfiltered. Deliberately treats sub-threshold examples as "
                          "too sparse to matter, trading away the 'always matches predict_word' "
                          "guarantee for those specific contexts.")
    args = ap.parse_args()

    n, counts, vocab, id_to_tok = load_model(args.model)
    print(f"loaded {args.model}: n={n}, vocab={len(vocab)}")
    print(f"\ntop-{args.top_k} table @ --min-count {args.min_count} "
          f"--max-prefix-len {args.max_prefix_len}"
          f"{' --strict' if args.strict else ''}:")

    # Validate build_streaming against build_topk_table+pack_one+build_trie (the small-scale,
    # already self-check-tested path) on a real slice of actual data before trusting it for
    # the expensive orders -- "same prediction as the ngram" has to hold on real data, not
    # just the synthetic demo() fixture.
    validate_k = 1 if n > 1 else 0  # order 0 is just the single global unigram context -- too
    # small a sample to mean anything; order 1 (ctx_len=1) has up to vocab-many real contexts
    validate_n = min(20000, len(counts[validate_k]))
    if validate_n:
        sample_ctx = list(counts[validate_k].keys())[:validate_n]
        sample = {c: dict(counts[validate_k][c]) for c in sample_ctx}  # copy, streaming pops its input
        ref_table = build_topk_table(sample, id_to_tok, args.min_count, args.max_prefix_len,
                                      args.top_k, strict=args.strict)
        ref_flat, _ = pack_one(ref_table)
        ref_trie_buf = encode_trie(build_trie(ref_table))
        sample2 = {c: dict(counts[validate_k][c]) for c in sample_ctx}
        got_flat, got_trie = build_streaming(sample2, id_to_tok, args.min_count,
                                              args.max_prefix_len, args.top_k, want_trie=True,
                                              strict=args.strict)
        got_trie_buf = encode_trie(got_trie)
        # both are packings of the same (ctx,prefix)->wids set, just in different row order --
        # compare as sets of decoded rows, not raw bytes; tries ARE directly byte-comparable
        # since a trie's key order comes from the shared-node structure, not encounter order
        assert set(_iter_flat_rows(ref_flat, validate_k)) == set(_iter_flat_rows(got_flat, validate_k)), \
            "build_streaming flat rows disagree with build_topk_table on real data"
        assert ref_trie_buf == got_trie_buf, \
            "build_streaming trie disagrees with build_trie on real data"
        print(f"validated build_streaming against build_topk_table on {validate_n:,} real "
              f"order-{validate_k} contexts: identical (ctx,prefix)->word predictions")

    # One order at a time, single streaming pass: pops the raw counts for that order as it
    # goes (input shrinks instead of coexisting with output) and packs bytes directly instead
    # of collecting row tuples first. The previous two-structure version (table dict + rows
    # list) hit 24-27GB RSS on order-4 at min-count=1 and was killed twice.
    flat_raw_total = flat_gz_total = 0
    trie_raw_total = trie_gz_total = 0
    for k in range(n):
        flat_buf, trie = build_streaming(counts[k], id_to_tok, args.min_count,
                                          args.max_prefix_len, args.top_k, args.trie,
                                          strict=args.strict)
        counts[k] = None

        flat_gz = gzip.compress(flat_buf, compresslevel=9)
        flat_raw_total += len(flat_buf)
        flat_gz_total += len(flat_gz)
        print(f"  order {k}: {len(flat_buf)/1e6:8.2f} MB raw flat, "
              f"{len(flat_gz)/1e6:8.2f} MB gzip flat")
        del flat_buf, flat_gz

        if args.trie:
            trie_buf = encode_trie(trie)
            del trie
            trie_gz = gzip.compress(trie_buf, compresslevel=9)
            trie_raw_total += len(trie_buf)
            trie_gz_total += len(trie_gz)
            print(f"           {len(trie_buf)/1e6:8.2f} MB raw trie, "
                  f"{len(trie_gz)/1e6:8.2f} MB gzip trie")
            del trie_buf, trie_gz

    print(f"\nTOTAL raw (flat)    : {flat_raw_total/1e6:8.2f} MB")
    print(f"TOTAL gzip -9 (flat): {flat_gz_total/1e6:8.2f} MB")
    if args.trie:
        print(f"TOTAL raw (trie)    : {trie_raw_total/1e6:8.2f} MB")
        print(f"TOTAL gzip -9 (trie): {trie_gz_total/1e6:8.2f} MB")

    print(f"\n(vocab table itself, id_to_tok strings: "
          f"{sum(len(t.encode())+1 for t in id_to_tok if t)/1e6:.2f} MB raw, not yet counted above)")


def demo():
    """Self-check: build_topk_table ranks by count descending, caps at top_k, still keys by
    every prefix length up to max_prefix_len, and drops sub-threshold candidates."""
    id_to_tok = {0: "star", 1: "stop", 2: "dog", 3: "stone"}
    d_ctx = {(2,): {0: 5, 1: 9, 3: 7}}  # after "dog": star=5, stop=9, stone=7

    out1 = build_topk_table(d_ctx, id_to_tok, min_count=1, max_prefix_len=1, top_k=1)
    assert out1 == {(2,): {"s": (1,)}}, out1  # prefix "s", top-1: stop(9) wins over star/stone

    out2 = build_topk_table(d_ctx, id_to_tok, min_count=1, max_prefix_len=1, top_k=3)
    assert out2 == {(2,): {"s": (1, 3, 0)}}, out2  # ranked stop(9) > stone(7) > star(5)

    out3 = build_topk_table(d_ctx, id_to_tok, min_count=1, max_prefix_len=3, top_k=2)
    assert out3[(2,)]["s"] == (1, 3)      # "s"   -> stop, stone (top-2 of all three)
    assert out3[(2,)]["st"] == (1, 3)     # "st"  -> stop, stone (star/stop/stone all match "st")
    assert out3[(2,)]["sto"] == (1, 3)    # "sto" -> stop, stone (star no longer matches "sto")
    assert out3[(2,)]["sta"] == (0,)      # "sta" -> star only (single candidate, not padded)

    # keep-intact exception: every candidate here is below min_count=10 (max is stop=9), so
    # the floor does NOT apply -- context keeps everyone, ranked, instead of being dropped
    out4 = build_topk_table(d_ctx, id_to_tok, min_count=10, max_prefix_len=1, top_k=3)
    assert out4 == {(2,): {"s": (1, 3, 0)}}, out4  # same ranking as out2, unaffected by min_count

    # mixed context: stop(15) clears min_count=10, so the floor DOES apply here -- star(3) and
    # stone(7) get filtered out same as normal, only stop survives
    mixed = {(2,): {0: 3, 1: 15, 3: 7}}
    out5 = build_topk_table(mixed, id_to_tok, min_count=10, max_prefix_len=1, top_k=3)
    assert out5 == {(2,): {"s": (1,)}}, out5

    # strict=True: no keep-intact exception -- the all-rare context (out4's case) is now
    # genuinely dropped instead of kept unfiltered
    out6 = build_topk_table(d_ctx, id_to_tok, min_count=10, max_prefix_len=1, top_k=3, strict=True)
    assert out6 == {}, out6
    # mixed context unaffected by strict (the floor already applied there either way)
    out7 = build_topk_table(mixed, id_to_tok, min_count=10, max_prefix_len=1, top_k=3, strict=True)
    assert out7 == {(2,): {"s": (1,)}}, out7

    # build_streaming must agree with build_topk_table on both cases (it pops its input, so
    # pass copies)
    stream_all_rare, _ = build_streaming(dict(d_ctx), id_to_tok, min_count=10, max_prefix_len=1,
                                          top_k=3, want_trie=False)
    assert set(_iter_flat_rows(stream_all_rare, ctx_len=1)) == \
        {((2,), "s", (1, 3, 0))}, list(_iter_flat_rows(stream_all_rare, ctx_len=1))
    stream_mixed, _ = build_streaming(dict(mixed), id_to_tok, min_count=10, max_prefix_len=1,
                                       top_k=3, want_trie=False)
    assert set(_iter_flat_rows(stream_mixed, ctx_len=1)) == \
        {((2,), "s", (1,))}, list(_iter_flat_rows(stream_mixed, ctx_len=1))

    # trie round-trip: encode out3, decode, confirm every lookup matches the source table
    trie = build_trie(out3)
    buf = encode_trie(trie)
    decoded, _ = decode_trie(buf)
    for prefix, expected in out3[(2,)].items():
        node = decoded
        for cid in (2,):
            node = node[cid]
        for ch in prefix:
            node = node[ch]
        assert node[VALUE_KEY] == expected, (prefix, node.get(VALUE_KEY), expected)
    print("demo ok")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        main()
