"""Authored fictional examples for UI demonstrations, not evaluation records."""
from __future__ import annotations
import random

EXAMPLES = {
    'SMS: prize offer': [
        'Congratulations! You have won a £200 cash prize. Call 0871 234 5678 within 24 hours to claim your reward.',
        'Your number was selected for a £500 shopping voucher. Visit https://rewards.example.com and enter your card number to collect it today.',
        'WINNER NOTICE: Claim your exclusive holiday prize before midnight. A £20 processing fee is required. Reply CLAIM for details.',
        'You have been shortlisted for our monthly cash draw. Call 0900 123 4567 now to activate your entry and confirm your bank account details.',
        'A free smartphone is waiting for you! Complete the delivery payment at https://gift.example.com within 2 hours to reserve it.',
        'Final reminder: your £1,000 prize remains unclaimed. Send your full name and account number to rewards@example.com to arrange payment.',
    ],
    'SMS: everyday message': [
        'Hey, are we still meeting at Jurong East at 7 tonight? I will bring the notes.',
        'I am at the library on level 3. Let me know when you arrive and I will come downstairs.',
        'Badminton is booked for Saturday at 4 pm. Please bring an extra shuttlecock if you have one.',
        'Could you pick up some milk on the way home? I already bought bread and eggs.',
        'The tutorial room has changed to LT4. Class still starts at 10 tomorrow morning.',
        'Thanks for helping with the slides. I have added my section and will review the figures after lunch.',
    ],
    'Email: account request': [
        'Dear customer, your account is locked. Visit https://account-check.example.com immediately and confirm your password to restore access.',
        'Subject: Account verification required\nWe detected unusual activity. Reply with your account number and one-time password within 24 hours to prevent suspension. Security Desk.',
        'Subject: Urgent mailbox notice\nYour mailbox will be disabled today. Confirm your login credentials at https://mail-verify.example.com to keep receiving messages.',
        'Dear account holder, a new device has accessed your profile. Call +65 6123 4567 now and provide your PIN so our verification team can secure your account.',
        'Subject: Final security reminder\nYour online access is pending review. Send your password to support@example.com before 6 pm or your account will remain locked.',
        'Dear customer, your annual identity check is incomplete. Upload your card number and security code at https://identity.example.com to avoid an interruption.',
    ],
    'Email: delivery notice': [
        'Subject: Parcel delivery failed\nYour package is on hold because £2.50 postage is unpaid. Enter your card number at https://parcel-pay.example.com today to arrange redelivery.',
        'Dear recipient, we could not confirm your delivery address. Pay the SGD 3 handling charge at https://shipping.example.com within 12 hours.',
        'Subject: Customs clearance required\nA parcel addressed to you requires a USD 15 release fee. Reply with your card number and billing address to release it immediately.',
        'Your shipment will be returned tomorrow unless you verify your payment information. Visit https://courier-check.example.com to complete the process.',
        'Subject: Express delivery reminder\nYour order is awaiting payment confirmation. Send your bank account details to delivery@example.com so we can schedule dispatch.',
        'We were unable to deliver your parcel today. Use https://reschedule.example.com and enter your password and card number to book a new slot.',
    ],
    'Email: invoice / payment request': [
        'Subject: Updated bank details\nPlease pay the outstanding £1,250 invoice to our new account today. Do not use the bank details on the previous invoice. Contact finance@example.com.',
        'Dear partner, the director has approved an urgent USD 2,000 transfer. Please buy gift cards and email the codes to purchases@example.com immediately.',
        'Subject: Payment overdue\nYour service will end within 24 hours. Pay SGD 88 using https://billing.example.com and send your card number to confirm.',
        'Hello, we have changed our settlement account. Please transfer the final project payment today and do not call the office because our lines are unavailable.',
        'Subject: Refund pending\nYou are eligible for a £340 refund. To receive it, reply with your bank account details and one-time password before midnight.',
        'A confidential supplier payment is required now. Kindly transfer USD 750 and send the confirmation to accounts@example.com without further approval.',
    ],
    'Email: everyday message': [
        'Subject: Project meeting\nHi team, our meeting is confirmed for Tuesday at 2 pm in the library. Please bring your experiment notes. Thanks!',
        'Subject: Draft review\nHello, I have attached the revised report draft. Could you review the introduction and leave comments before Friday?',
        'Subject: Room booking confirmation\nYour study room is reserved for 3 to 5 pm tomorrow. Please release the booking if your plans change.',
        'Subject: Seminar reminder\nThe NLP seminar starts at 11 am in LT2. Attendance is optional and the slides will be shared afterwards.',
        'Subject: Lunch next week\nHi Daniel, are you free for lunch on Wednesday? We can discuss the project timeline after class.',
        'Subject: Dataset documentation\nThanks for sending the README. I have checked the column names and will upload the cleaned charts this evening.',
    ],
    'Phone: suspicious dialogue': [
        'Caller: I am calling from your bank. There is an urgent problem with your account. Please tell me your one-time password now.\nCustomer: Why do you need my code?',
        'Caller: Your computer has been compromised. Install our remote support tool and read out the access code immediately.\nCustomer: I did not request technical support.',
        'Caller: You have an unpaid government fine. Transfer SGD 500 within one hour or further action will be taken.\nCustomer: Can I verify this through the official office?',
        'Caller: We are processing your refund. Please provide your card number and the code sent to your phone.\nCustomer: I have not applied for a refund.',
        'Caller: A delivery is waiting for you. Pay a small fee and tell me your banking PIN so we can authorise it.\nCustomer: Why would a courier need my PIN?',
        'Caller: This is your manager. Buy gift cards for a confidential payment and send the codes now.\nEmployee: I will call you back on the company number.',
    ],
    'Phone: everyday dialogue': [
        'Caller: Hi, I am outside the sports hall. Which entrance should I use?\nFriend: Use the entrance next to the cafe. I will meet you there.',
        'Caller: Hello, can we move our study session to 6 pm?\nFriend: Sure, the library works for me.',
        'Caller: Your table is ready. Are you still coming for dinner?\nCustomer: Yes, we will arrive in ten minutes.',
        'Caller: I am checking whether you received the revised slides.\nTeammate: Yes, I will review them after class.',
        'Caller: The maintenance visit is scheduled for tomorrow afternoon. Is that convenient?\nResident: Yes, please ring the doorbell when you arrive.',
        'Caller: Do you want me to collect the badminton rackets before practice?\nFriend: Yes, please. I will bring the shuttlecocks.',
    ],
}


def next_example(category: str, state: dict, state_key: str) -> str:
    """Draw the next shuffled example without repeating the previous one."""
    pool = EXAMPLES[category]
    key = f'{state_key}:{category}'
    record = state.get(key, {'remaining': [], 'last': None})
    remaining = list(record['remaining'])
    if not remaining:
        remaining = list(range(len(pool)))
        random.SystemRandom().shuffle(remaining)
        if len(pool) > 1 and remaining[-1] == record['last']:
            remaining[0], remaining[-1] = remaining[-1], remaining[0]
    index = remaining.pop()
    state[key] = {'remaining': remaining, 'last': index}
    return pool[index]
