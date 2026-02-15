package com.ap;

import java.security.PrivateKey;
import java.security.PublicKey;

public class Transaction {
    private String transactionId;
    private PublicKey sender;
    private PublicKey recipient;
    private float amount;
    private byte[] signature;

    private static int sequence = 0;

    public Transaction(PublicKey from, PublicKey to, float amount) {
        this.sender = from;
        this.recipient = to;
        this.amount = amount;
        this.transactionId = calculateHash();
    }

    private String calculateHash() {
        sequence++;
        return StringUtil.applySha256(
                StringUtil.getStringFromKey(sender) +
                StringUtil.getStringFromKey(recipient) +
                Float.toString(amount) +
                sequence
        );
    }

    public void generateSignature(PrivateKey privateKey) {
        String data = StringUtil.getStringFromKey(sender) + StringUtil.getStringFromKey(recipient) + Float.toString(amount);
        signature = StringUtil.applyECDSASig(privateKey, data);
    }

    public boolean verifySignature() {
        String data = StringUtil.getStringFromKey(sender) + StringUtil.getStringFromKey(recipient) + Float.toString(amount);
        return StringUtil.verifyECDSASig(sender, data, signature);
    }

    public boolean processTransaction(Blockchain blockchain) {
        if (verifySignature()) {
            System.out.println("Transaction signature verified successfully");
            return true;
        } else {
            System.out.println("Transaction signature failed to verify");
            return false;
        }
    }

    public String getTransactionId() {
        return transactionId;
    }

    public PublicKey getSender() {
        return sender;
    }

    public PublicKey getRecipient() {
        return recipient;
    }

    public float getAmount() {
        return amount;
    }

    public String getSenderAddress() {
        return sender != null ? StringUtil.getStringFromKey(sender) : null;
    }

    public String getRecipientAddress() {
        return StringUtil.getStringFromKey(recipient);
    }
}
