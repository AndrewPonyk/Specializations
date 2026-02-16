package com.ap;

import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;

// |su:11 WALLET: Stores your keys. Your public key is your address (share it), private key signs transactions (NEVER share it!)
public class Wallet {
    private PrivateKey privateKey;  // Keep secret! Anyone with this can spend your coins
    private PublicKey publicKey;  // This is your wallet address - share to receive coins
    private List<Transaction> transactions;

    public Wallet() {
        transactions = new ArrayList<>();
        generateKeyPair();
    }

    // |su:12 KEY GENERATION: Create ECDSA key pair (256-bit). Using BouncyCastle security provider.
    private void generateKeyPair() {
        try {
            KeyPairGenerator keyGen = KeyPairGenerator.getInstance("ECDSA", "BC");
            SecureRandom random = SecureRandom.getInstance("SHA1PRNG");
            keyGen.initialize(256, random);
            KeyPair keyPair = keyGen.generateKeyPair();
            this.privateKey = keyPair.getPrivate();
            this.publicKey = keyPair.getPublic();
        } catch (Exception e) {
            throw new RuntimeException("Error generating key pair: " + e.getMessage());
        }
    }

    public Transaction sendFunds(PublicKey recipient, float amount) {
        if (getBalance() < amount) {
            System.out.println("Not enough funds to complete transaction. Transaction discarded.");
            return null;
        }

        Transaction transaction = new Transaction(publicKey, recipient, amount);
        transaction.generateSignature(privateKey);
        transactions.add(transaction);
        return transaction;
    }

    public float getBalance() {
        float total = 0;
        for (Transaction tx : transactions) {
            if (tx.getSender() != null && tx.getSender().equals(publicKey)) {
                total -= tx.getAmount();
            }
        }
        return total;
    }

    public PublicKey getPublicKey() {
        return publicKey;
    }

    public PrivateKey getPrivateKey() {
        return privateKey;
    }

    public String getWalletAddress() {
        return StringUtil.getStringFromKey(publicKey);
    }

    public List<Transaction> getTransactions() {
        return transactions;
    }
}
