package com.ap;

import java.security.PrivateKey;
import java.security.PublicKey;

// |su:8 TRANSACTION: Transfer of funds between wallets. Signed with sender's private key to prove authenticity.
public class Transaction {
    private String transactionId;  // Unique hash identifying this transaction
    private PublicKey sender;  // |su:9 PUBLIC KEY CRYPTO: Public key = wallet address, private key = signing authority
    private PublicKey recipient;
    private float amount;
    private byte[] signature;  // |su:10 DIGITAL SIGNATURE: Proves sender authorized this transaction (ECDSA)

    private static int sequence = 0;
    private boolean isCoinbase;  // |su:14 COINBASE: Mining reward transaction (creates new coins, sender=null)

    public Transaction(PublicKey from, PublicKey to, float amount) {
        this.sender = from;
        this.recipient = to;
        this.amount = amount;
        this.isCoinbase = (from == null);
        this.transactionId = calculateHash();
    }

    // |su:15 COINBASE TX: Create mining reward transaction (null sender = new coins minted)
    public static Transaction createCoinbase(PublicKey minerAddress, float reward) {
        return new Transaction(null, minerAddress, reward);
    }

    private String calculateHash() {
        sequence++;
        String senderStr = (sender == null) ? "COINBASE" : StringUtil.getStringFromKey(sender);
        return StringUtil.applySha256(
                senderStr +
                        StringUtil.getStringFromKey(recipient) +
                        Float.toString(amount) +
                        sequence);
    }

    // Coinbase transactions don't need signatures (sender = null)
    public void generateSignature(PrivateKey privateKey) {
        if (isCoinbase) return;  // No signature for coinbase
        String data = StringUtil.getStringFromKey(sender) + StringUtil.getStringFromKey(recipient)
                + Float.toString(amount);
        signature = StringUtil.applyECDSASig(privateKey, data);
    }

    public boolean verifySignature() {
        if (isCoinbase) return true;  // Coinbase is always valid
        String data = StringUtil.getStringFromKey(sender) + StringUtil.getStringFromKey(recipient)
                + Float.toString(amount);
        return StringUtil.verifyECDSASig(sender, data, signature);
    }

    // |su:16 VERIFY: Coinbase transactions skip signature check (no sender = newly minted coins)
    public boolean processTransaction(Blockchain blockchain) {
        // Coinbase transactions (mining rewards) don't need signature verification
        if (isCoinbase) {
            System.out.println("Coinbase transaction: " + amount + " coins minted for miner");
            return true;
        }

        if (!verifySignature()) {
            System.out.println("Transaction signature failed to verify");
            return false;
        }

        if (getSender() != null) {
            float senderBalance = blockchain.getBalance(getSenderAddress());
            if (senderBalance < getAmount()) {
                System.out.println("Transaction failed: Not enough funds. Sender balance: " + senderBalance
                        + ", Amount: " + getAmount());
                return false;
            }
        }

        System.out.println("Transaction signature verified successfully");
        return true;
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

    public boolean isCoinbase() {
        return isCoinbase;
    }
}
