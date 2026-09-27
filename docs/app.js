/*
 * Loads the 4 per-order tries (order0..order3.trie.gz) exported by
 * scripts/build_release_assets.py and predicts the next word from (context, first letter),
 * matching infer_ngram.py's predict_word backoff exactly: try the longest available context
 * first (up to 3 preceding words), fall back to shorter contexts, then the unigram (order 0).
 *
 * Trie byte format (must match scripts/export_demo_model.py's encode_trie exactly):
 *   node := [hasValue:u8]
 *           [hasValue==1 ? nIds:u8, nIds * wordId:u32le : nothing]
 *           [nChildren:u32le]
 *           nChildren * child
 *   child := [tag:u8 (0=int ctx-id edge, 1=char edge)]
 *            [tag==0 ? key:u32le : keyLen:u8, keyLen bytes utf8]
 *            [childByteLen:u32le]
 *            [childByteLen bytes: the child node, recursively]
 *
 * The reader below never parses a whole trie into JS objects -- it walks the raw bytes
 * directly and skips any sibling subtree it doesn't need, via the length prefix. That's the
 * whole reason the format is length-prefixed rather than a flat sorted array.
 */

const ORDER_COUNT = 4; // n=4 model: orders 0 (unigram) .. 3 (4-gram, up to 3 words of context)
const MAX_CONTEXT = ORDER_COUNT - 1;

let orderBufs = null; // Uint8Array[4]
let idToTok = null;   // string[]
let tokToId = null;   // Map<string, number>

async function fetchGunzip(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch failed: ${url} (${res.status})`);
  const stream = res.body.pipeThrough(new DecompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

async function loadModel() {
  const [vocabBytes, ...bufs] = await Promise.all([
    fetchGunzip("data/vocab.txt.gz"),
    ...Array.from({ length: ORDER_COUNT }, (_, k) => fetchGunzip(`data/order${k}.trie.gz`)),
  ]);
  orderBufs = bufs;
  idToTok = new TextDecoder("utf-8").decode(vocabBytes).split("\n");
  tokToId = new Map();
  idToTok.forEach((tok, id) => {
    if (tok) tokToId.set(tok, id);
  });
}

// Read one node's header at `offset`: its word ids (if any) and where its children list starts.
function readNode(buf, offset) {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = offset;
  const hasValue = buf[pos];
  pos += 1;
  let wordIds = null;
  if (hasValue) {
    const nIds = buf[pos];
    pos += 1;
    wordIds = [];
    for (let i = 0; i < nIds; i++) {
      wordIds.push(view.getUint32(pos, true));
      pos += 4;
    }
  }
  return { wordIds, childrenOffset: pos };
}

// Linear-scan this node's children for an edge matching (wantTag, wantKey); skip every
// non-matching sibling's subtree via its length prefix instead of parsing it.
function matchChild(buf, childrenOffset, wantTag, wantKey) {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const decoder = new TextDecoder("utf-8");
  let pos = childrenOffset;
  const nChildren = view.getUint32(pos, true);
  pos += 4;
  for (let i = 0; i < nChildren; i++) {
    const tag = buf[pos];
    pos += 1;
    let key;
    if (tag === 0) {
      key = view.getUint32(pos, true);
      pos += 4;
    } else {
      const klen = buf[pos];
      pos += 1;
      key = decoder.decode(buf.subarray(pos, pos + klen));
      pos += klen;
    }
    const childLen = view.getUint32(pos, true);
    pos += 4;
    const childStart = pos;
    if (tag === wantTag && key === wantKey) return childStart;
    pos += childLen;
  }
  return -1;
}

// Descend ctxIds (word ids, oldest first) as int edges, then `letter` as a char edge.
// Returns the ranked word-id array at that (context, letter), or null if no such path exists.
function lookup(buf, ctxIds, letter) {
  let offset = 0;
  for (const cid of ctxIds) {
    const { childrenOffset } = readNode(buf, offset);
    const next = matchChild(buf, childrenOffset, 0, cid);
    if (next === -1) return null;
    offset = next;
  }
  const { childrenOffset } = readNode(buf, offset);
  const next = matchChild(buf, childrenOffset, 1, letter);
  if (next === -1) return null;
  return readNode(buf, next).wordIds;
}

// Same backoff order as predict_word/topk_by_letter in infer_ngram.py: longest context first,
// shrinking on a miss or an out-of-vocabulary word, down to the unigram as the final fallback.
function predict(contextWords, letter) {
  const recent = contextWords.slice(-MAX_CONTEXT);
  const ids = recent.map((w) => tokToId.get(w));
  for (let j = ids.length; j >= 1; j--) {
    const ctxIds = ids.slice(ids.length - j);
    if (ctxIds.some((id) => id === undefined)) continue;
    const hit = lookup(orderBufs[j], ctxIds, letter);
    if (hit) return hit.map((id) => idToTok[id]);
  }
  const hit0 = lookup(orderBufs[0], [], letter);
  return hit0 ? hit0.map((id) => idToTok[id]) : [];
}

// --- UI wiring ---

const typer = document.getElementById("typer");
const hint = document.getElementById("hint");
const chips = Array.from(document.querySelectorAll(".chip"));

function currentContextAndPrefix() {
  const value = typer.value;
  const endsWithSpace = /\s$/.test(value);
  const words = value.trim().split(/\s+/).filter(Boolean);
  if (endsWithSpace || words.length === 0) {
    return { context: words, prefix: "" };
  }
  return { context: words.slice(0, -1), prefix: words[words.length - 1] };
}

function renderChips(words) {
  chips.forEach((chip, i) => {
    const w = words[i];
    chip.textContent = w || "";
    chip.disabled = !w;
  });
}

function refresh() {
  const { context, prefix } = currentContextAndPrefix();
  if (!prefix) {
    renderChips([]);
    return;
  }
  const letter = prefix[0];
  const words = predict(context, letter);
  renderChips(words);
}

function insertSuggestion(word) {
  const { context } = currentContextAndPrefix();
  typer.value = [...context, word, ""].join(" ");
  typer.focus();
  refresh();
}

typer.addEventListener("input", () => {
  const pos = typer.selectionStart;
  typer.value = typer.value.toLowerCase();
  typer.setSelectionRange(pos, pos);
  refresh();
});

chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    if (!chip.disabled) insertSuggestion(chip.textContent);
  });
});

(async function init() {
  if (typeof DecompressionStream === "undefined") {
    hint.textContent = "Your browser doesn't support DecompressionStream -- try a recent Chrome, Firefox, or Safari.";
    typer.disabled = true;
    return;
  }
  hint.textContent = "Loading model…";
  try {
    await loadModel();
    hint.textContent = "Type a sentence — suggestions appear once you start a new word.";
    typer.disabled = false;
    typer.focus();
  } catch (err) {
    hint.textContent = "Failed to load the model: " + err.message;
  }
})();
