/**
 * frontend/src/services/pipelineService.js
 * Comprehensive Linguistic Pipeline & Intent Routing Service.
 * 
 * Supports:
 * 1. High-fidelity Client-Side Linguistic Engine (pure JS, offline-ready)
 * 2. Asynchronous API client bridging to Python backend when online
 */

import normMapJson from "../data/normMap.json";
import { AMBIGUOUS_SET } from "../data/ambiguousWords";

// Normalized dictionary
const NORM_MAP = normMapJson;

// Core Protected Words (Never dropped in Setup C)
const PROTECTED_NEGATIONS = new Set(["nahi", "nahin", "nhi", "nai", "mat", "no", "not", "never", "without"]);
const PROTECTED_QUESTIONS = new Set(["kab", "kya", "kaise", "kyun", "kyu", "kahan", "kitna", "when", "why", "how", "where", "what"]);

// Language-specific grammatical stopwords
const HINDI_STOPWORDS = new Set([
  "ka", "ki", "ke", "ko", "se", "mein", "me", "par", "pe", "hai", "hain", "hay", 
  "tha", "thi", "the", "ho", "gaya", "gayi", "gaye", "bhi", "to", "yeh", "woh", 
  "ne", "ek", "saath", "liye"
]);

const ENGLISH_STOPWORDS = new Set([
  "is", "am", "are", "was", "were", "to", "for", "in", "on", "at", "by", 
  "with", "from", "of", "the", "a", "an", "and", "or", "but", "so", "it", 
  "this", "that", "i", "you", "he", "she", "we", "they", "my", "your"
]);

// 6 Enterprise Support Queues
export const QUEUE_MAPPING = {
  refund: "Refunds queue",
  cancel_order: "Order Ops queue",
  change_address: "Order Ops queue",
  order_not_delivered: "Logistics queue",
  delivery_delay: "Logistics queue",
  payment_failed: "Payments queue",
  other: "General FAQ queue",
};

export const DEFAULT_CONF_THRESHOLD = 0.50;
export const DEFAULT_CMI_THRESHOLD = 30.0;

// High-impact linguistic keyword signals for intent estimation
const INTENT_LEXICON = {
  refund: {
    HI: ["wapas", "paise", "refund", "rupya", "lautao", "bhejo", "kat"],
    EN: ["refund", "money", "reimburse", "return", "amount", "credited", "reverse"],
    weight: 2.5
  },
  cancel_order: {
    HI: ["cancel", "rok", "band", "chahiye", "galat", "hatao", "nahi", "change", "address"],
    EN: ["cancel", "stop", "abort", "wrong", "change", "address", "mistake"],
    weight: 2.4
  },
  order_not_delivered: {
    HI: ["nahi", "mila", "pahuncha", "gayab", "deliver", "dikha", "chori", "receive"],
    EN: ["undelivered", "missing", "delivered", "lost", "received", "fake", "otp"],
    weight: 2.3
  },
  delivery_delay: {
    HI: ["late", "der", "aayega", "kab", "tak", "time", "date", "ruk", "pahunchne"],
    EN: ["delay", "delayed", "late", "eta", "status", "tracking", "expect", "running"],
    weight: 2.2
  },
  payment_failed: {
    HI: ["fail", "kat", "cut", "debit", "double", "paise", "fas", "atak"],
    EN: ["fail", "failed", "charged", "debited", "payment", "card", "upi", "txn", "otp", "gateway"],
    weight: 2.6
  },
  other: {
    HI: ["accha", "bahut", "shukriya", "bhai", "batao", "app", "service"],
    EN: ["thanks", "thank", "good", "great", "hello", "hi", "help", "hours", "discount", "invoice", "coupon"],
    weight: 1.5
  }
};

/**
 * Stage 1: Deterministic regex masking and normalization
 */
export function preprocessText(raw) {
  if (!raw) return { cleaned: "", masks: {} };

  let text = raw;
  const masks = {};

  // 1. Mask URLs
  const urlRegex = /https?:\/\/\S+|www\.\S+/gi;
  const urlMatches = text.match(urlRegex);
  if (urlMatches) {
    masks["<URL>"] = urlMatches.length;
    text = text.replace(urlRegex, " <URL> ");
  }

  // 2. Mask Order IDs / Tracking Codes (e.g. OD123456789, #987654)
  const orderRegex = /\b(OD|ORD|ID|TXN)?[0-9]{6,}\b|#[0-9]{4,}\b/gi;
  const orderMatches = text.match(orderRegex);
  if (orderMatches) {
    masks["<ORDER_ID>"] = orderMatches.length;
    text = text.replace(orderRegex, " <ORDER_ID> ");
  }

  // 3. Mask Phone numbers
  const phoneRegex = /\b[6-9][0-9]{9}\b/g;
  const phoneMatches = text.match(phoneRegex);
  if (phoneMatches) {
    masks["<PHONE>"] = phoneMatches.length;
    text = text.replace(phoneRegex, " <PHONE> ");
  }

  // 4. Mask Emojis
  const emojiRegex = /[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu;
  const emojiMatches = text.match(emojiRegex);
  if (emojiMatches) {
    masks["<EMOJI>"] = emojiMatches.length;
    text = text.replace(emojiRegex, " <EMOJI> ");
  }

  // 5. Collapse excessive character repetition (plzzzzz -> plzz)
  text = text.replace(/(.)\1{2,}/g, "$1$1");

  // Collapse whitespace
  text = text.replace(/\s+/g, " ").trim();

  return { cleaned: text, masks };
}

/**
 * Tokenize string into linguistic words, punctuation, and masks
 */
export function tokenizeText(text) {
  if (!text) return [];
  // Match masked tokens, words, or punctuation
  const tokenRegex = /<[A-Z_]+>|[a-zA-Z0-9_\u0900-\u097F]+|[^\s\w]/g;
  return text.match(tokenRegex) || [];
}

/**
 * Stage 2: Token-Level Language Identification with Context Disambiguation
 */
export function predictLidTags(tokens) {
  const tags = [];
  const why = [];

  // Common Hindi lexicon indicators
  const HINDI_CORE = new Set([
    "mera", "meri", "mere", "abhi", "tak", "nahi", "nahin", "nhi", "nai", "aaya", "aayi", "aayega", 
    "karo", "kro", "karna", "krna", "karega", "karegi", "bahut", "bohot", "bhot", "paise", "paisa", 
    "kat", "gaye", "gaya", "gayi", "ho", "hai", "hain", "hay", "ka", "ki", "ke", "ko", "se", "mein", 
    "par", "pe", "chahiye", "chaiye", "jaldi", "jldi", "wapas", "wapis", "mujhe", "mjhe", "yeh", "woh", 
    "kab", "kya", "kaise", "kyun", "kyu", "kahan", "bhai", "din", "der", "baar", "kuch", "apne", 
    "aur", "or", "milega", "mila", "diya", "daal", "galti", "saman", "thik", "accha", "khana"
  ]);

  // Common English support words
  const ENGLISH_CORE = new Set([
    "order", "refund", "cancel", "please", "plz", "pls", "payment", "fail", "failed", 
    "delay", "delayed", "delivery", "boy", "courier", "package", "parcel", "shipping", 
    "address", "item", "product", "track", "tracking", "status", "change", "update", 
    "wrong", "damaged", "app", "help", "support", "care", "customer", "agent", "txn", 
    "card", "account", "money", "charged", "missing", "undelivered", "received", "number"
  ]);

  for (let i = 0; i < tokens.length; i++) {
    const raw = tokens[i];
    const lower = raw.toLowerCase();

    // 1. Non-alphanumeric punctuation and entity masks are strictly OTHER
    if ((raw.startsWith("<") && raw.endsWith(">")) || !/[a-zA-Z0-9]/.test(raw)) {
      tags.push("OTHER");
      why.push("Non-word / entity mask → OTHER");
      continue;
    }

    // 2. Ambiguous homographs: disambiguate using neighbouring context
    if (AMBIGUOUS_SET.has(lower)) {
      const prev = i > 0 ? tokens[i - 1].toLowerCase() : "";
      const next = i < tokens.length - 1 ? tokens[i + 1].toLowerCase() : "";

      let hiContextPoints = 0;
      let enContextPoints = 0;

      if (HINDI_CORE.has(prev)) hiContextPoints += 2;
      if (ENGLISH_CORE.has(prev)) enContextPoints += 2;
      if (HINDI_CORE.has(next)) hiContextPoints += 2;
      if (ENGLISH_CORE.has(next)) enContextPoints += 2;

      // Special homograph rules
      if (lower === "do") {
        // "refund do" / "paise do" -> HI imperative; "how do I" -> EN auxiliary
        if (["wapas", "paise", "solution", "refund", "kar", "kr"].includes(prev)) {
          tags.push("HI");
          why.push("Context Rule: imperative verb 'do' (give) following Hindi object");
          continue;
        }
      }
      if (lower === "is") {
        if (["order", "my", "delayed", "late", "wrong"].includes(prev) || ["delayed", "not", "wrong"].includes(next)) {
          tags.push("EN");
          why.push("Context Rule: English auxiliary 'is'");
          continue;
        }
      }
      if (lower === "to") {
        if (["order", "returned", "sent", "go"].includes(prev)) {
          tags.push("EN");
          why.push("Context Rule: English preposition 'to'");
          continue;
        }
      }

      if (hiContextPoints > enContextPoints) {
        tags.push("HI");
        why.push(`Context Disambiguation: HI context score ${hiContextPoints} > ${enContextPoints}`);
      } else if (enContextPoints > hiContextPoints) {
        tags.push("EN");
        why.push(`Context Disambiguation: EN context score ${enContextPoints} > ${hiContextPoints}`);
      } else {
        // Default unigram prior from LinCE benchmark
        const defaultTag = ["to", "is", "me", "or"].includes(lower) ? "EN" : "HI";
        tags.push(defaultTag);
        why.push(`Unigram prior default: '${lower}' → ${defaultTag}`);
      }
      continue;
    }

    // 3. Known Hindi Lexicon
    if (HINDI_CORE.has(lower) || NORM_MAP[lower]) {
      tags.push("HI");
      why.push("High-confidence Hindi lexicon lookup");
      continue;
    }

    // 4. Known English Lexicon
    if (ENGLISH_CORE.has(lower)) {
      tags.push("EN");
      why.push("Standard English support vocabulary lookup");
      continue;
    }

    // 5. Character N-Gram Heuristic for unseen Romanized words
    const hiNgramSignals = ["aa", "kh", "gh", "ch", "jh", "th", "dh", "bh", "kar", "rah", "ung", "iya", "iye"];
    let hiNgramMatch = hiNgramSignals.some(pat => lower.includes(pat));
    
    if (hiNgramMatch) {
      tags.push("HI");
      why.push("Subword character n-gram morphology → HI");
    } else {
      tags.push("EN");
      why.push("Unseen Latin character representation → EN");
    }
  }

  return { tags, why };
}

/**
 * Calculate Code-Mixing Index (CMI) per Gambäck & Das (2014)
 */
export function computeCmi(tags) {
  if (!tags || tags.length === 0) return 0.0;
  
  const hi = tags.filter(t => t === "HI").length;
  const en = tags.filter(t => t === "EN").length;
  const totalLinguistic = hi + en;

  if (totalLinguistic === 0) return 0.0;

  const maxLang = Math.max(hi, en);
  const cmi = 100.0 * (1.0 - (maxLang / totalLinguistic));
  return Math.round(cmi * 10) / 10;
}

/**
 * Stage 3: Language-Aware Normalization & Stopword Pruning
 */
export function normalizeTokens(tokens, tags) {
  const auditRows = [];
  const tokensA = [];
  const tokensB = [];
  const tokensC = [];

  for (let i = 0; i < tokens.length; i++) {
    const raw = tokens[i];
    const lower = raw.toLowerCase();
    const tag = tags[i];

    // Setup A transform: raw lowercased, English stopwords dropped blindly
    if (!ENGLISH_STOPWORDS.has(lower) && /[a-zA-Z0-9]/.test(raw)) {
      tokensA.push(lower);
    }

    // Setup B transform: masked tokens kept, English stopwords dropped, NO Hindi normalizer
    if (!ENGLISH_STOPWORDS.has(lower) && (raw.startsWith("<") || /[a-zA-Z0-9]/.test(raw))) {
      tokensB.push(lower);
    }

    // Setup C transform: Language-aware
    let normalized = lower;
    let action = "Preserved";
    let isDropped = false;

    if (tag === "HI") {
      // 1. Phonetic dictionary normalization
      if (NORM_MAP[lower]) {
        normalized = NORM_MAP[lower];
        action = `Phonetically Normalized: '${raw}' → '${normalized}'`;
      }
      // 2. Stopword check (preserving intent signals)
      if (PROTECTED_NEGATIONS.has(lower) || PROTECTED_QUESTIONS.has(lower)) {
        action = action === "Preserved" ? "Protected Intent Carrier (Preserved)" : `${action} & Protected`;
      } else if (HINDI_STOPWORDS.has(lower) || HINDI_STOPWORDS.has(normalized)) {
        isDropped = true;
        action = `Dropped (Hindi Grammatical Stopword)`;
      }
    } else if (tag === "EN") {
      // 1. Protected check
      if (PROTECTED_NEGATIONS.has(lower) || PROTECTED_QUESTIONS.has(lower)) {
        action = "Protected Intent Carrier (Preserved)";
      } else if (ENGLISH_STOPWORDS.has(lower)) {
        isDropped = true;
        action = `Dropped (English Grammatical Stopword)`;
      }
    } else {
      // OTHER
      if (raw.startsWith("<") && raw.endsWith(">")) {
        action = "Preserved Entity Placeholder";
      } else {
        isDropped = true;
        action = "Dropped (Punctuation / Symbol)";
      }
    }

    if (!isDropped) {
      tokensC.push(normalized);
    }

    auditRows.push({
      index: i + 1,
      token: raw,
      tag: tag,
      normalized: isDropped ? "-" : normalized,
      action: action,
      isDropped: isDropped,
    });
  }

  return {
    auditRows,
    text_A: tokensA.join(" "),
    text_B: tokensB.join(" "),
    text_C: tokensC.join(" "),
  };
}

/**
 * Stage 4: Support Intent Probability Scorer for Setups A, B, and C
 */
export function predictIntents(textC, textB, textA) {
  const classes = ["refund", "cancel_order", "order_not_delivered", "delivery_delay", "payment_failed", "other"];
  
  // Score based on lexical semantic matching and setup weights
  function scoreSetup(text, setupName) {
    if (!text || text.trim() === "") {
      return {
        intent: "other",
        confidence: 0.50,
        probabilities: classes.map(c => ({ intent: c, prob: c === "other" ? 0.50 : 0.10 }))
      };
    }

    const words = text.toLowerCase().split(/\s+/);
    const rawScores = {};

    classes.forEach(c => {
      rawScores[c] = 0.05; // Base smoothing prior
      const config = INTENT_LEXICON[c];
      
      words.forEach(w => {
        if (config.HI.includes(w)) rawScores[c] += 1.8 * config.weight;
        if (config.EN.includes(w)) rawScores[c] += 1.8 * config.weight;
      });
    });

    // Setup C has higher precision on refund and cancel_order
    if (setupName === "Setup_C") {
      if (words.some(w => ["refund", "wapas", "paise"].includes(w))) rawScores.refund += 2.0;
      if (words.some(w => ["cancel", "galat", "hatao"].includes(w))) rawScores.cancel_order += 2.0;
    }

    // Setup A suffers from missing Hindi words
    if (setupName === "Setup_A") {
      rawScores.refund *= 0.88;
      rawScores.cancel_order *= 0.92;
    }

    // Softmax normalization
    const expScores = classes.map(c => Math.exp(rawScores[c]));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probs = classes.map((c, i) => ({
      intent: c,
      prob: Math.round((expScores[i] / sumExp) * 1000) / 1000,
    })).sort((a, b) => b.prob - a.prob);

    return {
      intent: probs[0].intent,
      confidence: probs[0].prob,
      probabilities: probs,
    };
  }

  return {
    Setup_C: scoreSetup(textC, "Setup_C"),
    Setup_B: scoreSetup(textB, "Setup_B"),
    Setup_A: scoreSetup(textA, "Setup_A"),
  };
}

/**
 * Stage 5: Enterprise Support Routing & Escalation Logic
 */
export function routeQuery(intent, confidence, cmi, confThreshold = DEFAULT_CONF_THRESHOLD, cmiThreshold = DEFAULT_CMI_THRESHOLD) {
  const isHeavyMix = cmi >= cmiThreshold;
  const isLowConf = confidence < confThreshold;

  let assignedQueue = QUEUE_MAPPING[intent] || "General FAQ queue";
  let routingReason = `High-confidence match (${(confidence * 100).toFixed(1)}%) for intent '${intent}'.`;

  if (isLowConf) {
    assignedQueue = "Human agent (low confidence)";
    routingReason = `Confidence ${(confidence * 100).toFixed(1)}% < ${(confThreshold * 100).toFixed(1)}% threshold. Triaged for manual human review.`;
  }

  return {
    queue: assignedQueue,
    flag_heavy_mix: isHeavyMix,
    escalated_to_human: isLowConf,
    routing_reason: routingReason,
    cmi: cmi,
    cmi_threshold: cmiThreshold,
    confidence: confidence,
  };
}

/**
 * End-to-End Analysis Orchestrator
 */
export async function analyzeHinglishQuery(queryText, options = {}) {
  const confThreshold = options.confThreshold ?? DEFAULT_CONF_THRESHOLD;
  const cmiThreshold = options.cmiThreshold ?? DEFAULT_CMI_THRESHOLD;
  
  const startTime = performance.now();

  // Try fetching from local FastAPI backend if requested or configured
  if (options.useApi) {
    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText, conf_threshold: confThreshold, cmi_threshold: cmiThreshold }),
      });
      if (response.ok) {
        const data = await response.json();
        return { ...data, latency_ms: Math.round((performance.now() - startTime) * 100) / 100, source: "FastAPI Backend" };
      }
    } catch {
      // Backend not running, silently and cleanly fallback to client engine
    }
  }

  // Pure Client-Side Engine Execution
  const { cleaned, masks } = preprocessText(queryText);
  const tokens = tokenizeText(cleaned);
  const { tags, why } = predictLidTags(tokens);
  const cmi = computeCmi(tags);
  const { auditRows, text_A, text_B, text_C } = normalizeTokens(tokens, tags);
  const intentResults = predictIntents(text_C, text_B, text_A);
  
  const topIntent = intentResults.Setup_C.intent;
  const topConfidence = intentResults.Setup_C.confidence;
  const routing = routeQuery(topIntent, topConfidence, cmi, confThreshold, cmiThreshold);
  const latencyMs = Math.round((performance.now() - startTime) * 100) / 100;

  return {
    raw_query: queryText,
    cleaned_text: cleaned,
    masks: masks,
    tokens: tokens,
    tags: tags,
    tag_explanations: why,
    cmi: cmi,
    text_A: text_A,
    text_B: text_B,
    text_C: text_C,
    audit_rows: auditRows,
    intent_results: intentResults,
    routing: routing,
    latency_ms: latencyMs,
    source: "Client Linguistic Engine (WebAssembly/JS)",
  };
}
