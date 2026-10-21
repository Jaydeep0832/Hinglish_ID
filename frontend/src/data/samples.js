/**
 * 7 Pre-defined Benchmark Queries matching app/demo.py
 */
export const SAMPLE_QUERIES = [
  {
    id: 1,
    label: "Sample 1: Refund Dispute (Mixed)",
    shortLabel: "Refund Dispute",
    text: "Mera order abhi tak nahi aaya, please refund karo",
    expected: "refund (High Grievance)",
    category: "refund",
  },
  {
    id: 2,
    label: "Sample 2: Payment Deducted (Emoji + Order ID)",
    shortLabel: "Payment Deducted",
    text: "payment fail ho gaya but paise kat gaye 😡 OD123456789",
    expected: "payment_failed",
    category: "payment_failed",
  },
  {
    id: 3,
    label: "Sample 3: Fast Cancellation (Hindi Heavy)",
    shortLabel: "Fast Cancellation",
    text: "order cancel karna hai plz jaldi, galat item order ho gaya",
    expected: "cancel_order",
    category: "cancel_order",
  },
  {
    id: 4,
    label: "Sample 4: Delivery Delay (English Heavy + URL)",
    shortLabel: "Delivery Delay",
    text: "My order is delayed by 4 days, kab tak aayega? https://track.delivery.in/OD998",
    expected: "delivery_delay",
    category: "delivery_delay",
  },
  {
    id: 5,
    label: "Sample 5: Address Modification (Code-Mixed)",
    shortLabel: "Address Modification",
    text: "mujhe shipping address change karwana hai naya flat number add karo",
    expected: "cancel_order / Order Ops",
    category: "change_address",
  },
  {
    id: 6,
    label: "Sample 6: Undelivered Order (Courier Issue)",
    shortLabel: "Undelivered Order",
    text: "4 din ho gaye parcel nahi mila, delivery boy phone pick nahi kar raha",
    expected: "order_not_delivered",
    category: "order_not_delivered",
  },
  {
    id: 7,
    label: "Sample 7: Positive Feedback (General FAQ)",
    shortLabel: "Positive Feedback",
    text: "Thanks a lot, customer care agent was very helpful app accha hai",
    expected: "other",
    category: "other",
  },
];
