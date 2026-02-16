package com.ap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Block {
    // |su:1 BLOCK: A container for transactions. Each block contains the hash of the previous block, creating the "chain".
    private String hash;
    private String previousHash;  // Links to previous block - this is what makes it a blockchain!
    private long timestamp;
    private int nonce;  // |su:2 PROOF OF WORK: A counter incremented during mining to find a valid hash
    private List<Transaction> transactions;
    private String merkleRoot;  // |su:3 MERKLE TREE: Root hash of all transactions - efficient verification
    private Map<String, Float> pendingBalances;  // Track balances within this block during tx addition

    public Block(String previousHash) {
        this.previousHash = previousHash;
        this.timestamp = System.currentTimeMillis();
        this.nonce = 0;
        this.transactions = new ArrayList<>();
        this.merkleRoot = "";
        this.pendingBalances = new HashMap<>();
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

    // |su:4 MINING: Find hash starting with N zeros. Higher difficulty = more computational work required
    public void mineBlock(int difficulty) {
        merkleRoot = StringUtil.getMerkleRoot(transactions);
        String target = new String(new char[difficulty]).replace('\0', '0');
        while (!hash.substring(0, difficulty).equals(target)) {
            nonce++;
            hash = calculateHash();
            //System.err.println("None:" + nonce + " Hash:" + hash);
        }
    }

    public boolean addTransaction(Transaction transaction, Blockchain blockchain) {
        if (transaction == null) return false;

        if (!previousHash.equals("0")) {
            // Use pending balances (includes earlier txs in this block)
            float currentBalance = getPendingBalance(blockchain, transaction.getSenderAddress());

            if (!transaction.verifySignature()) {
                System.out.println("Transaction signature failed to verify");
                return false;
            }

            // Check balance for non-coinbase transactions
            if (!transaction.isCoinbase()) {
                if (currentBalance < transaction.getAmount()) {
                    System.out.println("Transaction failed: Not enough funds. Balance: " + currentBalance + ", Amount: " + transaction.getAmount());
                    return false;
                }
                // Deduct from sender's pending balance
                pendingBalances.put(transaction.getSenderAddress(), currentBalance - transaction.getAmount());
            } else {
                // Coinbase: add to recipient's pending balance
                float recipientBalance = getPendingBalance(blockchain, transaction.getRecipientAddress());
                pendingBalances.put(transaction.getRecipientAddress(), recipientBalance + transaction.getAmount());
                System.out.println("Coinbase transaction: " + transaction.getAmount() + " coins minted for miner");
            }

            System.out.println("Transaction signature verified successfully");
        }

        transactions.add(transaction);
        System.out.println("Transaction added to block: " + transaction.getTransactionId());
        return true;
    }

    // Get balance from blockchain + pending txs in this block
    private float getPendingBalance(Blockchain blockchain, String address) {
        if (address == null) return 0f;
        float chainBalance = blockchain.getBalance(address);
        float pendingDelta = pendingBalances.getOrDefault(address, 0f);
        return chainBalance + pendingDelta;
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
