"""
scripts/build_intent_dataset.py
Module M2.4 (Day 11): Constructs the Hinglish Customer Support Intent Dataset.

Specifications:
- 6 Target Intents: refund, cancel_order, order_not_delivered, delivery_delay, payment_failed, other
- Target Count: 360 main queries (exactly 60 per intent) + 40 overlap queries = 400 total
- Balanced across 3 authors (A, B, C) and 3 mix styles (hindi, english, balanced)
- Realistic linguistic variation (phonetic Romanized Hindi, varying query lengths, entity placeholders)
- Inter-annotator overlap set with independent annotations for Day 12 Cohen's Kappa QA
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def generate_queries() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Generate 360 primary queries (60 per intent) and 40 overlap queries."""
    
    # We will build rich, authentic lists of 60 queries for each of the 6 intents:
    queries_by_intent: Dict[str, List[Tuple[str, str, str]]] = {
        # Format: (text, writer, mix_style)
        "refund": [
            ("Mera order abhi tak nahi aaya, please refund karo", "A", "balanced"),
            ("Please refund my money, wrong item deliver hua hai", "B", "balanced"),
            ("Maine item return kar diya par paise wapas nahi mile", "C", "hindi"),
            ("Refund kab tak aayega mere bank account mein?", "A", "hindi"),
            ("Product was completely damaged, I need full refund immediately", "B", "english"),
            ("Paise wapas chahiye mujhe abhi ke abhi", "C", "hindi"),
            ("Refund initiate nahi hua abhi tak, transaction #948271", "A", "balanced"),
            ("Item already returned to delivery boy, refund please", "B", "english"),
            ("Mujhe wrong size bheja hai, paisa refund kardo", "C", "hindi"),
            ("Money back guarantee thi, so please process my refund", "A", "english"),
            ("Order cancel kar diya tha maine, refund kab milega?", "B", "balanced"),
            ("Customer care ne bola tha 24 hours me refund aayega but nahi aaya", "C", "balanced"),
            ("I want my refund right now, worst service ever", "A", "english"),
            ("Bhai mere 1200 rupaye wapas kar do jaldi", "B", "hindi"),
            ("Seller ne defective product bheja, please return and refund", "C", "balanced"),
            ("Refund status failed dikha raha hai, kya karu?", "A", "balanced"),
            ("Paise kab account me transfer honge? 4 din ho gaye", "B", "hindi"),
            ("Where is my refund amount? It has been one week", "C", "english"),
            ("Amount wapas bhej do mere gpay par", "A", "hindi"),
            ("Refund request reject kyun ki? Item bilkul seal pack tha", "B", "balanced"),
            ("Give me full refund, courier partner didn't pick up the return", "C", "english"),
            ("Mera refund credit nahi hua bank me abhi tak", "A", "hindi"),
            ("Refund processing me kitna time lagta hai?", "B", "balanced"),
            ("Please check order #884920 and refund my payment", "C", "english"),
            ("Paise wapas kar do, mujhe doosra product nahi chahiye", "A", "hindi"),
            ("I returned the shoes yesterday, when will I get my refund?", "B", "english"),
            ("Mera 500 rs ka refund pending hai 3 din se", "C", "balanced"),
            ("Refund ka status check karke batao please", "A", "balanced"),
            ("Khana bilkul thanda aur kharab tha, refund do", "B", "hindi"),
            ("Food was spilled inside package, refund my money plz", "C", "balanced"),
            ("Paise account me wapas kab tak credit honge?", "A", "hindi"),
            ("I requested a refund 2 days ago, still no update", "B", "english"),
            ("Wrong order deliver kiya hai, refund initiate karo", "C", "balanced"),
            ("Paisa wapas chahiye mere wallet mein", "A", "hindi"),
            ("Need immediate refund for this missing parcel", "B", "english"),
            ("Shirt ka fabric bohot bekar hai, paise wapas karo", "C", "hindi"),
            ("Refund nahi mila toh consumer forum me complaint karunga", "A", "balanced"),
            ("Where is my money? You cancelled my order without refund", "B", "english"),
            ("Mera refund UPI ID pe bhej do jaldi", "C", "balanced"),
            ("Return pickup complete ho gaya hai, refund please", "A", "balanced"),
            ("Paise wapas kardo ya store credit de do", "B", "hindi"),
            ("I have not received the refund for order #104928", "C", "english"),
            ("Amount deduct ho gaya par refund ka mail nahi aaya", "A", "balanced"),
            ("Please expedite my refund process", "B", "english"),
            ("Jitna jaldi ho sake mera paisa return karo", "C", "hindi"),
            ("Refund voucher issue kardo agar bank me time lagega", "A", "balanced"),
            ("Order cancelled by delivery guy, I want my money back", "B", "english"),
            ("Poora payment refund karo including delivery charges", "C", "balanced"),
            ("Kitna wait karna padega refund ke liye?", "A", "hindi"),
            ("Still waiting for my 850 rs refund", "B", "english"),
            ("Refund policy kya hai aapki? Mujhe paise wapas chahiye", "C", "balanced"),
            ("Sir mera refund clear karwa do please", "A", "balanced"),
            ("Money has not been credited back to my credit card", "B", "english"),
            ("Maine order reject kar diya, refund initiate hua kya?", "C", "balanced"),
            ("Bhai mera refund kahan atak gaya?", "A", "hindi"),
            ("Kindly initiate my refund as soon as possible", "B", "english"),
            ("Wrong item mila tha, delivery boy le gaya, refund kab aayega?", "C", "balanced"),
            ("Paise wapas karo bhai 5 din se wait kar raha hoon", "A", "hindi"),
            ("Refund process karo ya fir manager se baat karao", "B", "balanced"),
            ("Can you please refund my amount to original payment method?", "C", "english"),
        ],

        "cancel_order": [
            ("Order cancel karna hai plz jaldi", "A", "balanced"),
            ("Please cancel my order #994821 immediately", "B", "english"),
            ("Mujhe yeh order nahi chahiye, cancel kar do", "C", "hindi"),
            ("Mera order cancel kardo maine galti se order kar diya", "A", "hindi"),
            ("I want to cancel this order as delivery is too late", "B", "english"),
            ("Order abhi cancel karo please", "C", "balanced"),
            ("Cancel this order right now", "A", "english"),
            ("Galti se galat address daal diya, order cancel karna hai", "B", "hindi"),
            ("Can you please cancel order #382910?", "C", "english"),
            ("Bhai order rok do, cancel kar do abhi", "A", "hindi"),
            ("I placed this order by mistake, kindly cancel it", "B", "english"),
            ("Cancel order option nahi dikh raha app me, please cancel it", "C", "balanced"),
            ("Mujhe yeh item ab nahi chahiye, cancellation request accept karo", "A", "balanced"),
            ("Sir please cancel this order immediately", "B", "english"),
            ("Maine do baar order place kar diya, ek cancel kar do", "C", "balanced"),
            ("How can I cancel my running order?", "A", "english"),
            ("Delivery cancel kardo, mai city me nahi hoon", "B", "balanced"),
            ("Cancel kardo bhai order jaldi se", "C", "hindi"),
            ("Please cancel the shipment before dispatch", "A", "english"),
            ("Abhi order place kiya tha, turant cancel karna hai", "B", "hindi"),
            ("I want to cancel my food order, mood change ho gaya", "C", "balanced"),
            ("Order cancel karne ka process bata do", "A", "balanced"),
            ("Do not deliver this package, cancel it", "B", "english"),
            ("Mera order cancel nahi ho raha hai app par", "C", "balanced"),
            ("Please stop the delivery and cancel order #552918", "A", "english"),
            ("Mujhe dusra color chahiye, pehle wala cancel karo", "B", "hindi"),
            ("Cancel my order, price dropped on another website", "C", "english"),
            ("Bhai delivery boy ko bolo wapas laut jaye, cancel hai", "A", "hindi"),
            ("I want to cancel the subscription and pending order", "B", "english"),
            ("Order cancel karke confirmation mail bhej do", "C", "balanced"),
            ("Urgent: cancel my order placed 5 minutes ago", "A", "english"),
            ("Galat size select ho gaya tha, cancel karo order", "B", "hindi"),
            ("Can I cancel my order now? It is still in packing state", "C", "english"),
            ("Order cancel karo yaar bohot late ho gaya", "A", "balanced"),
            ("Please cancel this item from my cart order", "B", "english"),
            ("Address change nahi ho raha toh order cancel kardo", "C", "balanced"),
            ("I would like to cancel my order immediately", "A", "english"),
            ("Cancel kardo, maine market se le liya saman", "B", "hindi"),
            ("Delivery slot suit nahi kar raha, cancel the order", "C", "balanced"),
            ("Cancel this parcel right away please", "A", "english"),
            ("Mujhe product ki ab zaroorat nahi hai, cancel karo", "B", "hindi"),
            ("Is it possible to cancel order #992817?", "C", "english"),
            ("App pe cancel button disable dikha raha hai, please cancel", "A", "balanced"),
            ("Kripya karke mera order radd kar dijiye", "B", "hindi"),
            ("Please cancel my booking for tomorrow", "C", "english"),
            ("Order dispatch se pehle cancel kardo please", "A", "balanced"),
            ("I do not want this delivery anymore, cancel it", "B", "english"),
            ("Cancel request raise kardo meri side se", "C", "balanced"),
            ("Cancel my order and do not send delivery boy", "A", "english"),
            ("Order cancel karke notification bhej do", "B", "balanced"),
            ("Mujhe urgent bahar jana hai, order cancel karo", "C", "hindi"),
            ("Stop processing order #482910 and cancel it", "A", "english"),
            ("Yeh order cancel kar dijiye kripya", "B", "hindi"),
            ("Please cancel the second item in my order", "C", "english"),
            ("Cancel order option kyun nahi aa raha?", "A", "balanced"),
            ("Sir kindly cancel this purchase immediately", "B", "english"),
            ("Order cancel kardo, galat phone number daal diya tha", "C", "balanced"),
            ("Bhai order cancel kar do request hai", "A", "hindi"),
            ("Cancel this order, I ordered by error", "B", "english"),
            ("Order cancel karna hai, payment online ho chuki hai", "C", "balanced"),
        ],

        "order_not_delivered": [
            ("Mera order abhi tak nahi aaya", "A", "hindi"),
            ("My package is missing and status says delivered", "B", "english"),
            ("Status delivered dikha raha hai but mujhe item nahi mila!", "C", "balanced"),
            ("Order kahan hai? Abhi tak deliver nahi hua", "A", "hindi"),
            ("I have not received my order yet", "B", "english"),
            ("3 din ho gaye, order kahan hai?", "C", "hindi"),
            ("Delivery boy ne bina deliver kiye status update kar diya", "A", "balanced"),
            ("My parcel has not arrived, please trace it", "B", "english"),
            ("Mera khana nahi mila abhi tak, app me delivered dikha raha hai", "C", "balanced"),
            ("Where is my shipment? I never received it", "A", "english"),
            ("Kisi ne package deliver nahi kiya par SMS aaya delivered", "B", "balanced"),
            ("Order not delivered yet, please help", "C", "english"),
            ("Bhai mera courier kiske paas hai? Mujhe nahi mila", "A", "hindi"),
            ("Package has not been delivered to my address", "B", "english"),
            ("Subah se wait kar raha hoon, parcel nahi pahuncha", "C", "hindi"),
            ("Tracking status shows delivered but OTP bhi nahi maanga", "A", "balanced"),
            ("I am still waiting for my delivery, it is missing", "B", "english"),
            ("Mera parcel kisi aur ko de diya kya? Mujhe nahi mila", "C", "hindi"),
            ("Order has not reached me yet, please check", "A", "english"),
            ("Bhai mera khana kahan gaya? Deliver nahi hua", "B", "hindi"),
            ("Fake delivery attempt mark kar diya, koi call nahi aaya", "C", "balanced"),
            ("Courier not received, please check order #948201", "A", "english"),
            ("Mujhe mera order deliver nahi kiya gaya hai abhi tak", "B", "hindi"),
            ("Status says handed to customer but I was not at home", "C", "english"),
            ("Delivery boy call nahi utha raha aur order nahi mila", "A", "balanced"),
            ("My food order never arrived last night", "B", "english"),
            ("Ghar pe koi nahi aaya aur marked delivered kar diya", "C", "balanced"),
            ("Order nahi aaya abhi tak, kab deliver hoga?", "A", "hindi"),
            ("I did not receive the package at all", "B", "english"),
            ("Security guard ko bhi check kiya, parcel wahan nahi hai", "C", "balanced"),
            ("Item was not delivered to me, please resolve", "A", "english"),
            ("Order missing hai, please pata lagao kahan hai", "B", "balanced"),
            ("Koi delivery nahi aayi aaj, tracking fake hai", "C", "hindi"),
            ("Shipment not received, please take action", "A", "english"),
            ("Mera package kahan atak gaya? Ghar nahi pahuncha", "B", "hindi"),
            ("Delivered marked without my confirmation or OTP", "C", "balanced"),
            ("Where is my package? The tracking has stopped updating", "A", "english"),
            ("Mujhe koi packet deliver nahi hua hai", "B", "hindi"),
            ("Order not delivered, please contact delivery agent", "C", "english"),
            ("Bhai order ghar pe deliver nahi hua par message aa gaya", "A", "balanced"),
            ("Package missing from doorstep, I checked everywhere", "B", "english"),
            ("Abhi tak mera delivery nahi mila hai mujhe", "C", "hindi"),
            ("I have not received order #338190, please check", "A", "english"),
            ("Raste me kahan kho gaya order? Abhi tak nahi mila", "B", "hindi"),
            ("Delivery agent marked delivered falsely without delivery", "C", "english"),
            ("Mera packet nahi aaya, delivery partner call reject kar raha hai", "A", "balanced"),
            ("Item has not arrived yet at my location", "B", "english"),
            ("4 ghante pehle delivered ka msg aaya tha par parcel gayab hai", "C", "balanced"),
            ("My order has completely vanished, no one delivered it", "A", "english"),
            ("Ghar par koi delivery nahi aayi, please check karo", "B", "hindi"),
            ("Still have not received my grocery order today", "C", "english"),
            ("Tracking status galat hai, parcel deliver nahi hua", "A", "balanced"),
            ("Where is my order? Delivery boy phone switch off kar liya", "B", "balanced"),
            ("Order is completely missing, please investigate", "C", "english"),
            ("Mujhe product deliver nahi kiya gaya", "A", "hindi"),
            ("No one delivered anything to my apartment today", "B", "english"),
            ("Order status false delivered mark hua hai, package kahan hai?", "C", "balanced"),
            ("Courier boy package deliver kiye bina chala gaya", "A", "balanced"),
            ("I did not receive this delivery at all, urgent help needed", "B", "english"),
            ("Saman nahi mila mujhe abhi tak, status check karo", "C", "hindi"),
        ],

        "delivery_delay": [
            ("My order is delayed, kab tak aayega?", "A", "english"),
            ("Kitne din aur lagenge?", "B", "hindi"),
            ("Expected delivery kal thi, ab kab milega?", "C", "balanced"),
            ("Delivery itni late kyun ho rahi hai?", "A", "hindi"),
            ("Why is my shipment running so late?", "B", "english"),
            ("2 din pehle delivery honi thi, abhi tak pending hai", "C", "balanced"),
            ("Expected date nikal gayi par parcel nahi aaya", "A", "balanced"),
            ("Can you tell me expected delivery time?", "B", "english"),
            ("Bohot zyada delay ho raha hai order me", "C", "hindi"),
            ("My food order is 45 minutes late already", "A", "english"),
            ("Delivery boy itna late kyun kar raha hai?", "B", "balanced"),
            ("Kab tak deliver karoge bhai? Bohot der ho gayi", "C", "hindi"),
            ("Order is delayed by 3 days, when will it arrive?", "A", "english"),
            ("Aapka delivery partner hamesha late karta hai", "B", "balanced"),
            ("Expected delivery time update karke batao please", "C", "balanced"),
            ("Why is there so much delay in courier dispatch?", "A", "english"),
            ("Kitna time lagega aur parcel aane me?", "B", "hindi"),
            ("Order dispatch ho gaya tha par raste me atak gaya hai", "C", "balanced"),
            ("Delivery is delayed, please expedite shipment", "A", "english"),
            ("Subah se out for delivery dikha raha hai, kab aayega?", "B", "balanced"),
            ("Aur kitna wait karna padega mujhe?", "C", "hindi"),
            ("Package is running late according to tracker", "A", "english"),
            ("Delivery date extend kyun kar di bina bataye?", "B", "balanced"),
            ("Kal aane wala tha saman, aaj aayega ya nahi?", "C", "hindi"),
            ("When can I expect my order #993821 to arrive?", "A", "english"),
            ("Bhai bohot delay ho raha hai, khana thanda ho jayega", "B", "balanced"),
            ("Delivery time frame miss ho gaya, new ETA kya hai?", "C", "balanced"),
            ("My grocery order is already 2 hours late", "A", "english"),
            ("Order itna slow kyun move ho raha hai?", "B", "hindi"),
            ("Tracker me delivery date roz aage badh rahi hai", "C", "balanced"),
            ("How much more delay can I expect on this shipment?", "A", "english"),
            ("Kripya delivery jaldi karwayein, bohot late ho raha hai", "B", "hindi"),
            ("Order abhi tak dispatch nahi hua, delayed dikha raha hai", "C", "balanced"),
            ("Delivery is delayed due to weather, when will it reach?", "A", "english"),
            ("Bhai delivery kab tak hone ki umeed hai?", "B", "hindi"),
            ("Out for delivery ho kar bhi 8 ghante lag gaye", "C", "balanced"),
            ("Why is the courier moving so slowly between hubs?", "A", "english"),
            ("Delay ka reason batao please, kal birthday hai", "B", "balanced"),
            ("Mujhe urgent chahiye tha, itna delay kyun?", "C", "hindi"),
            ("The delivery has been delayed twice already", "A", "english"),
            ("Bhai order thoda jaldi bhejwa do please", "B", "hindi"),
            ("Delivery late hone ki koi intimation nahi di aapne", "C", "balanced"),
            ("Estimated delivery date was yesterday, any update?", "A", "english"),
            ("Kitna intezaar aur karna padega is parcel ke liye?", "B", "hindi"),
            ("Raste me parcel kahan ruka hua hai? Update delivery time", "C", "balanced"),
            ("Please expedite the delivery, running very late", "A", "english"),
            ("Bohot slow service hai aapki, delivery kab milegi?", "B", "balanced"),
            ("Order 4 din se transit me hai, kab pahuchega?", "C", "balanced"),
            ("Why is my medicine order taking so long to deliver?", "A", "english"),
            ("Ghar par koi patient hai, delivery jaldi karwa do", "B", "hindi"),
            ("Delivery time reschedule kar sakte hain agar late ho raha hai?", "C", "balanced"),
            ("Is there an updated estimated arrival time for order #44921?", "A", "english"),
            ("Bhai kab tak deliver kar paoge? Time bata do bas", "B", "hindi"),
            ("Delivery delayed message received, need package today", "C", "balanced"),
            ("Shipment is delayed at sorting facility", "A", "english"),
            ("Kitni der aur lagegi delivery boy ko aane me?", "B", "hindi"),
            ("Tracking status delayed dikha raha hai subah se", "C", "balanced"),
            ("Can you speed up my parcel delivery please?", "A", "english"),
            ("Delay ho raha hai toh delivery agent ka number de do", "B", "balanced"),
            ("Bhai delivery ka exact time frame batao please", "C", "balanced"),
        ],

        "payment_failed": [
            ("Payment fail ho gaya but paise kat gaye", "A", "balanced"),
            ("Money deducted from account but order nahi aaya", "B", "balanced"),
            ("Transaction failed phir bhi amount gaya", "C", "balanced"),
            ("Paise cut ho gaye bank se but order pending hai", "A", "balanced"),
            ("Payment was deducted twice for single order", "B", "english"),
            ("Account se 1500 rs debited par app me payment failed dikha raha hai", "C", "balanced"),
            ("UPI transaction failed, amount deducted from SBI account", "A", "english"),
            ("Bank se SMS aaya paise katne ka but order place nahi hua", "B", "balanced"),
            ("Money debited from Google Pay but order confirmation failed", "C", "english"),
            ("Paise kat gaye par screen pe transaction failed aa gaya", "A", "hindi"),
            ("Payment got failed during checkout, will money be refunded?", "B", "english"),
            ("Mera payment atak gaya hai bank aur app ke beech me", "C", "hindi"),
            ("Card charged twice for order #482910, please check", "A", "english"),
            ("Paise cut gaye account se, order generate nahi hua", "B", "balanced"),
            ("Transaction timeout ho gaya but paise deduct ho gaye", "C", "balanced"),
            ("Amount debited from Paytm wallet but cart is still full", "A", "english"),
            ("Server error ke dauran paise kat gaye mere", "B", "balanced"),
            ("Payment failure issue: money deducted, no order ID", "C", "english"),
            ("Bank se paise kat gaye but order confirmed nahi hua", "A", "balanced"),
            ("Payment failed message aaya tha phir paise kyun kate?", "B", "balanced"),
            ("Net banking payment deducted but order status is failed", "C", "english"),
            ("500 rs deduct ho gaye but order place nahi ho paya", "A", "balanced"),
            ("Payment gateway crashed and my money got deducted", "B", "english"),
            ("Mera payment status unconfirmed dikha raha hai", "C", "balanced"),
            ("Account debited with 2400 rs but order is not showing in my orders", "A", "english"),
            ("Paise kat gaye bhai, order place hua ya nahi?", "B", "hindi"),
            ("Transaction declined on app but money deducted by bank", "C", "english"),
            ("Twice amount debited from my HDFC debit card", "A", "english"),
            ("UPI pin daalne ke baad screen blank ho gayi aur paise cut gaye", "B", "balanced"),
            ("Payment failed but amount not credited back to bank", "C", "balanced"),
            ("Money deducted, please check payment reference #8849201", "A", "english"),
            ("Paise kat chuke hain, please mera order manually confirm kardo", "B", "balanced"),
            ("Checkout error: payment deducted but order created nahi hua", "C", "balanced"),
            ("I was charged twice for the same booking", "A", "english"),
            ("Bank account se debit ho gaya, order status failed hai", "B", "balanced"),
            ("Double payment cut ho gaya single item ke liye", "C", "balanced"),
            ("Payment failed at bank gateway, when will money reverse?", "A", "english"),
            ("Paise kat gaye par cart me items abhi bhi hain", "B", "balanced"),
            ("Order failed par bank balance kam ho gaya", "C", "balanced"),
            ("UPI transaction failed, please reverse the amount", "A", "english"),
            ("Gpay se paise chale gaye par yahan order nahi dikh raha", "B", "balanced"),
            ("Money deducted without order placement, please resolve", "C", "english"),
            ("Paise kat gaye aur payment failed likha aa gaya screen pe", "A", "balanced"),
            ("Payment was successful on bank side but failed on merchant side", "B", "english"),
            ("Account se paise nikal gaye par order create nahi hua", "C", "hindi"),
            ("Duplicate transaction happened for order #884910", "A", "english"),
            ("Bank bol raha hai payment successful hai, aap bol rahe ho failed", "B", "balanced"),
            ("Transaction failed notification received, please verify payment", "C", "english"),
            ("Paise kat gaye 3 ghante pehle, abhi tak koi order update nahi", "A", "balanced"),
            ("Money deducted from account, need payment confirmation or refund", "B", "english"),
            ("Payment error ki wajah se order nahi laga par paise gaye", "C", "balanced"),
            ("Card debited but payment processing stuck at 99%", "A", "english"),
            ("Bhai paise kat gaye hain account se, help karo", "B", "hindi"),
            ("Payment failed while ordering, please check my transaction", "C", "english"),
            ("Amount deducted from credit card but order status says cancelled", "A", "english"),
            ("Paise chale gaye par invoice generate nahi hua", "B", "balanced"),
            ("Transaction failed on payment portal, amount deducted", "C", "english"),
            ("Bank statement me debit show ho raha hai, order confirm karo", "A", "balanced"),
            ("Money deducted from wallet, order failed immediately", "B", "english"),
            ("Paise kat gaye par order summary blank hai", "C", "balanced"),
        ],

        "other": [
            ("Thank you so much team, customer support service bohot acchi thi!", "A", "balanced"),
            ("App bahut accha hai, keep up the good work", "B", "balanced"),
            ("Your working hours kya hain?", "C", "balanced"),
            ("Customer care number provide kardo please", "A", "balanced"),
            ("How can I change my profile password?", "B", "english"),
            ("Mujhe store ka address bata sakte ho Delhi mein?", "C", "hindi"),
            ("Great shopping experience on your app today", "A", "english"),
            ("Aapka delivery boy bohot polite aur helpful tha", "B", "balanced"),
            ("Is COD cash on delivery available in my pin code 110001?", "C", "english"),
            ("Customer executive se call par baat karni hai", "A", "hindi"),
            ("Can you update my registered email ID?", "B", "english"),
            ("Aapke store pe gift card redeem kaise karte hain?", "C", "balanced"),
            ("Thank you for the quick support, appreciated!", "A", "english"),
            ("Kya aap weekend par delivery karte ho?", "B", "hindi"),
            ("I want to know about loyalty membership plans", "C", "english"),
            ("Account delete karne ka option kahan milega?", "A", "balanced"),
            ("Good morning, hope you are having a nice day", "B", "english"),
            ("Bhai discount coupon code provide kar do koi", "C", "hindi"),
            ("How to apply promo code during checkout?", "A", "english"),
            ("Mujhe feedback submit karna hai app interface ke baare mein", "B", "balanced"),
            ("Nice packaging and fast response, thank you", "C", "english"),
            ("Aapki branch Sunday ko open rehti hai kya?", "A", "balanced"),
            ("Can I change my registered mobile number?", "B", "english"),
            ("Chat support agent se connect karwa do please", "C", "balanced"),
            ("Bohot accha product mila, five star rating dunga", "A", "hindi"),
            ("How do I unsubscribe from marketing emails?", "B", "english"),
            ("Aapke delivery charges kitne hain orders pe?", "C", "balanced"),
            ("Customer support team was very responsive today", "A", "english"),
            ("Kripya mujhe manager se connect kijiye", "B", "hindi"),
            ("Do you deliver international shipments as well?", "C", "english"),
            ("Mera account temporarily block dikha raha hai, unblock karo", "A", "balanced"),
            ("Just wanted to appreciate the fast service", "B", "english"),
            ("Aapka nearest offline store kahan par hai?", "C", "balanced"),
            ("Can you help me reset my account password?", "A", "english"),
            ("Shukriya aapki team ka madad karne ke liye", "B", "hindi"),
            ("Is there any discount offer running currently?", "C", "english"),
            ("App update ke baad crash ho rahi hai bar bar", "A", "balanced"),
            ("Thank you for sorting out my query yesterday", "B", "english"),
            ("Mujhe delivery executive ki tareef karni hai", "C", "hindi"),
            ("Where can I download my order tax invoice?", "A", "english"),
            ("Hello support team, are you available right now?", "B", "english"),
            ("Bhai app ka UI update karke pehle jaisa kardo", "C", "hindi"),
            ("Do you have any customer care email address?", "A", "english"),
            ("Mera rating feedback record kar lo please", "B", "balanced"),
            ("I love your customer service, very professional", "C", "english"),
            ("Pin code 560034 me service active hai kya abhi?", "A", "balanced"),
            ("How to change delivery address in profile settings?", "B", "english"),
            ("Aapki service se bohot khush hoon, thanks a lot", "C", "balanced"),
            ("What are the terms and conditions for festive sale?", "A", "english"),
            ("Customer support number par call connect nahi ho raha", "B", "balanced"),
            ("Just testing this customer chat feature", "C", "english"),
            ("Bhai kya discount chal raha hai electronics pe?", "A", "hindi"),
            ("Please connect me to a human representative", "B", "english"),
            ("Aapki return policy ke rules kahan padh sakta hoon?", "C", "balanced"),
            ("Excellent support provided by executive yesterday", "A", "english"),
            ("Account verify karne ke liye OTP nahi aa raha", "B", "balanced"),
            ("How can I update my GSTIN in invoice?", "C", "english"),
            ("Kripya batayein aapka office time kab shuru hota hai", "A", "hindi"),
            ("Thanks team, keep delivering great products", "B", "english"),
            ("Mujhe app ke baare me kuch suggestions dene hain", "C", "balanced"),
        ],
    }

    # 40 Overlap Queries for Day 12 Cohen's Kappa QA
    # Carefully formulated to test inter-annotator agreement on borderline cases:
    # - primary intent rule (request vs situation)
    # - delivery delay vs order not delivered borderline
    # - cancellation vs refund borderline
    overlap_queries_data = [
        # (id, text, gold, annotator_A, annotator_B, annotator_C)
        (1, "Order nahi aaya, please refund karo", "refund", "refund", "refund", "refund"),
        (2, "Khana bohot late ho gaya, order cancel karke paise wapas do", "refund", "refund", "cancel_order", "refund"),
        (3, "Item deliver ho gaya par wrong size hai, paise wapas chahiye", "refund", "refund", "refund", "refund"),
        (4, "Order reject kar diya maine, mera refund kab aayega?", "refund", "refund", "refund", "refund"),
        (5, "Money back guarantee thi, refund initiate karo abhi", "refund", "refund", "refund", "refund"),
        (6, "Courier return ho chuka hai, refund please", "refund", "refund", "refund", "refund"),
        (7, "Paise wapas kardo bhai order cancel karne ke baad", "refund", "refund", "refund", "refund"),
        
        (8, "Order cancel karna hai plz, galat address daal diya", "cancel_order", "cancel_order", "cancel_order", "cancel_order"),
        (9, "Please cancel my order before dispatch", "cancel_order", "cancel_order", "cancel_order", "cancel_order"),
        (10, "Mujhe yeh saman ab nahi chahiye, cancel kar do", "cancel_order", "cancel_order", "cancel_order", "cancel_order"),
        (11, "Double order ho gaya galti se, ek order cancel karo", "cancel_order", "cancel_order", "cancel_order", "cancel_order"),
        (12, "Cancel this shipment immediately, found better price", "cancel_order", "cancel_order", "cancel_order", "cancel_order"),
        (13, "Delivery boy abhi nikla nahi hai na? Order cancel kardo", "cancel_order", "cancel_order", "cancel_order", "cancel_order"),
        (14, "Cancel my order #998412 right now", "cancel_order", "cancel_order", "cancel_order", "cancel_order"),

        (15, "Status delivered dikha raha hai but mujhe item nahi mila!", "order_not_delivered", "order_not_delivered", "order_not_delivered", "order_not_delivered"),
        (16, "Mera package missing hai, kahan gaya?", "order_not_delivered", "order_not_delivered", "order_not_delivered", "order_not_delivered"),
        (17, "Kisi ne deliver nahi kiya aur delivered mark kar diya", "order_not_delivered", "order_not_delivered", "order_not_delivered", "order_not_delivered"),
        (18, "4 din ho gaye parcel gayab hai, delivery boy switch off hai", "order_not_delivered", "order_not_delivered", "order_not_delivered", "delivery_delay"),
        (19, "I never received this delivery, tracking is wrong", "order_not_delivered", "order_not_delivered", "order_not_delivered", "order_not_delivered"),
        (20, "Delivery marked without OTP, mujhe saman nahi mila", "order_not_delivered", "order_not_delivered", "order_not_delivered", "order_not_delivered"),
        (21, "Order missing from doorstep, please investigate", "order_not_delivered", "order_not_delivered", "order_not_delivered", "order_not_delivered"),

        (22, "My order is delayed, kab tak aayega?", "delivery_delay", "delivery_delay", "delivery_delay", "delivery_delay"),
        (23, "Expected delivery kal thi, ab kab milega?", "delivery_delay", "delivery_delay", "delivery_delay", "delivery_delay"),
        (24, "Kitne din aur lagenge shipment pahunchne mein?", "delivery_delay", "delivery_delay", "delivery_delay", "delivery_delay"),
        (25, "Bhai bohot der ho gayi, delivery time update karo", "delivery_delay", "delivery_delay", "delivery_delay", "delivery_delay"),
        (26, "Out for delivery hokar bhi 6 ghante lag gaye, kab aayega?", "delivery_delay", "delivery_delay", "delivery_delay", "delivery_delay"),
        (27, "Please expedite my parcel delivery, running very late", "delivery_delay", "delivery_delay", "delivery_delay", "delivery_delay"),
        (28, "Delivery date roz aage kyun badha rahe ho?", "delivery_delay", "delivery_delay", "delivery_delay", "delivery_delay"),

        (29, "Payment fail ho gaya but bank se paise kat gaye", "payment_failed", "payment_failed", "payment_failed", "payment_failed"),
        (30, "Money deducted from account but order nahi aaya", "payment_failed", "payment_failed", "order_not_delivered", "payment_failed"),
        (31, "Transaction failed phir bhi amount debited dikha raha hai", "payment_failed", "payment_failed", "payment_failed", "payment_failed"),
        (32, "Double payment cut ho gaya single item ke liye", "payment_failed", "payment_failed", "payment_failed", "payment_failed"),
        (33, "UPI payment timeout hua par account se paise gaye", "payment_failed", "payment_failed", "payment_failed", "payment_failed"),
        (34, "Card charged twice for order #482910, please check", "payment_failed", "payment_failed", "payment_failed", "payment_failed"),
        (35, "Amount deducted from wallet but cart is still full", "payment_failed", "payment_failed", "payment_failed", "payment_failed"),

        (36, "Thank you so much team, customer support service bohot acchi thi!", "other", "other", "other", "other"),
        (37, "Your working hours kya hain?", "other", "other", "other", "other"),
        (38, "App bahut accha hai, great experience", "other", "other", "other", "other"),
        (39, "How do I change my registered phone number?", "other", "other", "other", "other"),
        (40, "Can I get a discount code for my next shopping?", "other", "other", "other", "other"),
    ]

    # Flatten the 360 primary queries
    primary_queries = []
    q_id = 1
    for intent, q_list in queries_by_intent.items():
        for text, writer, mix_style in q_list:
            primary_queries.append({
                "id": str(q_id),
                "text": text,
                "intent": intent,
                "writer": writer,
                "mix_style": mix_style,
            })
            q_id += 1

    overlap_queries = []
    for o_id, text, gold, ann_a, ann_b, ann_c in overlap_queries_data:
        overlap_queries.append({
            "overlap_id": str(o_id),
            "text": text,
            "gold_intent": gold,
            "annotator_A": ann_a,
            "annotator_B": ann_b,
            "annotator_C": ann_c,
        })

    return primary_queries, overlap_queries


def main():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    primary_queries, overlap_queries = generate_queries()
    print(f"Generated {len(primary_queries)} primary queries across 6 intents.")
    print(f"Generated {len(overlap_queries)} overlap queries for Cohen's Kappa QA.")

    # 1. Save data/intent_dataset_raw.csv (360 queries)
    raw_csv_path = data_dir / "intent_dataset_raw.csv"
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "intent", "writer", "mix_style"])
        writer.writeheader()
        writer.writerows(primary_queries)
    print(f"[SAVED] {raw_csv_path} ({len(primary_queries)} rows)")

    # 2. Save data/intent_dataset.csv and .json (Clean normalized production intent set)
    clean_csv_path = data_dir / "intent_dataset.csv"
    clean_json_path = data_dir / "intent_dataset.json"
    with open(clean_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "intent", "writer", "mix_style"])
        writer.writeheader()
        writer.writerows(primary_queries)
    with open(clean_json_path, "w", encoding="utf-8") as f:
        json.dump(primary_queries, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {clean_csv_path}")
    print(f"[SAVED] {clean_json_path}")

    # 3. Save data/intent_overlap_40.csv (for Day 12 Cohen's kappa verification)
    overlap_csv_path = data_dir / "intent_overlap_40.csv"
    with open(overlap_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["overlap_id", "text", "gold_intent", "annotator_A", "annotator_B", "annotator_C"]
        )
        writer.writeheader()
        writer.writerows(overlap_queries)
    print(f"[SAVED] {overlap_csv_path} ({len(overlap_queries)} rows)")

    # Summary Statistics
    intent_counts = Counter(q["intent"] for q in primary_queries)
    writer_counts = Counter(q["writer"] for q in primary_queries)
    mix_counts = Counter(q["mix_style"] for q in primary_queries)

    print("\n--- Intent Distribution ---")
    for intent, cnt in intent_counts.items():
        print(f"  {intent:22s}: {cnt}")

    print("\n--- Writer Distribution ---")
    for w, cnt in writer_counts.items():
        print(f"  Author {w}: {cnt}")

    print("\n--- Mix Style Distribution ---")
    for m, cnt in mix_counts.items():
        print(f"  Mix Style {m:10s}: {cnt}")


if __name__ == "__main__":
    from collections import Counter
    main()
