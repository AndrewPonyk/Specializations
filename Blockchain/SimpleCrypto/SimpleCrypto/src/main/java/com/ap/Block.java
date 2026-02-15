package com.ap;

import java.util.ArrayList;
import java.util.List;

public class Block {
    private String hash;
    private String previousHash;
    private long timestamp;
    private int nonce;
    private List<Transaction> transactions;
    private String merkleRoot;

    public Block(String previousHash) {
        this.previousHash = previousHash;
        this.timestamp = System.currentTimeMillis();
        this.nonce = 0;
        this.transactions = new ArrayList<>();
        this.merkleRoot = "";
        this.hash = calculateHash();
    }

    public String calculateHash() {
        return StringUtil.applySha256(
                previousHash +
                Long.toString(timestamp) +
                Integer.toString(nonce) +
                merkleRoot
        );
    }

    public void mineBlock(int difficulty) {
        merkleRoot = StringUtil.getMerkleRoot(transactions);
        String target = new String(new char[difficulty]).replace('\0', '0');
        while (!hash.substring(0, difficulty).equals(target)) {
            nonce++;
            hash = calculateHash();
        }
    }

    public boolean addTransaction(Transaction transaction, Blockchain blockchain) {
        if (transaction == null) return false;

        if (!previousHash.equals("0")) {
            if (!transaction.processTransaction(blockchain)) {
                System.out.println("Transaction failed to process. Discarded.");
                return false;
            }
        }

        transactions.add(transaction);
        System.out.println("Transaction added to block: " + transaction.getTransactionId());
        return true;
    }

    public String getHash() {
        return hash;
    }

    public String getPreviousHash() {
        return previousHash;
    }

    public List<Transaction> getTransactions() {
        return transactions;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public int getNonce() {
        return nonce;
    }

    public void setHash(String hash) {
        this.hash = hash;
    }
}
