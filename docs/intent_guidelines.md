# Intent Annotation Guidelines v1 — HinglishID-Router
# Owner: Member B | Date: [Day 7]

## Overview

Annotate each Hinglish customer support query with ONE intent label.
Use the **primary-intent rule**: if a query expresses multiple intents,
label it as the action the customer is *requesting* (not mentioning).

Example:
> "Order nahi aaya, please refund karo"
> → label = `refund` (requested action), NOT `order_not_delivered` (mentioned situation)

---

## The 6 Intents

### 1. `refund`
Customer explicitly wants money back for a delivered/cancelled/wrong item.

**Positive signals:** refund, wapas karo, paisa wapas, money back, return money
**Examples:**
- "Mujhe refund chahiye, wrong item aaya"
- "Please give me my money back, order cancel ho gaya"
- "Paisa wapas karo yaar"

**Borderline:** If customer says "cancel karo" AND "paisa wapas do" → label `refund`

---

### 2. `cancel_order`
Customer wants to cancel a pending/processing order (before delivery).

**Positive signals:** cancel, band karo (order context), cancel karna hai
**Examples:**
- "Order cancel karna hai plz"
- "Mera order cancel kar do abhi"
- "Cancel my order immediately"

**Borderline:** If item already delivered and customer wants return → `refund`

---

### 3. `order_not_delivered`
Customer reporting that order has not arrived / is missing.

**Positive signals:** nahi aaya, missing, nahi mila, where is my order, kahan hai
**Examples:**
- "Mera order abhi tak nahi aaya"
- "My package is missing"
- "3 din ho gaye, order kahan hai?"

**Borderline:** If delivery is simply late → `delivery_delay`. If they want refund → `refund`.

---

### 4. `delivery_delay`
Customer complaining that delivery is late but has not yet declared it missing.

**Positive signals:** late, delayed, kab aayega, expected date miss, wait kar raha hoon
**Examples:**
- "My order is delayed, kab tak aayega?"
- "Kitne din aur lagenge?"
- "Expected delivery kal thi, ab kab milega?"

**Borderline:** If 5+ days past expected date and customer angry → could be `order_not_delivered`

---

### 5. `payment_failed`
Customer reporting payment deducted but order not placed / failed transaction.

**Positive signals:** payment fail, paise kat gaye, transaction failed, amount debited
**Examples:**
- "Payment fail ho gaya but paise kat gaye"
- "Money deducted, order nahi aaya"
- "Transaction failed phir bhi amount gaya"

**Borderline:** If order placed successfully and customer wants refund → `refund`

---

### 6. `other`
Does not fit any of the above 5 intents.

**Examples:**
- Compliments: "App bahut accha hai"
- General questions: "Your working hours kya hain?"
- Gibberish / test queries
- Feedback not related to a specific order

---

## Annotation Rules

1. **One label per query** — no multi-label
2. **Primary intent** — what is the customer requesting, not just mentioning
3. **No guessing** — if genuinely unclear after 30 seconds, write `other` + note
4. **Variety** — write queries with different: mix ratios, spellings, lengths, tone
5. **writer column** — always fill your team member ID (A/B/C)
6. **mix_style** — `hindi` (>70% HI), `english` (>70% EN), `balanced` (otherwise)

## Quality Targets
- ≥ 60 examples per intent
- Cohen's kappa ≥ 0.6 between two annotators on 40-50 overlap set
- <10% near-duplicates after dedup

---

## CSV Format

```csv
id,text,intent,writer,mix_style
1,"Mera order abhi tak nahi aaya, please refund karo",refund,A,balanced
2,"payment fail ho gaya but paise kat gaye",payment_failed,B,balanced
3,"order cancel karna hai plz jaldi",cancel_order,C,hindi
4,"My order is delayed, kab tak aayega?",delivery_delay,A,english
```
