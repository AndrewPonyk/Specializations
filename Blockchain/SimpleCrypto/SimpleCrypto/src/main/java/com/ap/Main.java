package com.ap;

import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.SecureRandom;

// |su:13 DEMO: This main() demonstrates the full flow: create blockchain → wallets → transactions → mining → validation
public class Main {
    public static void main(String[] args) {
        System.out.println("=== Simple Cryptocurrency Demo ===\n");

        try {
            // Initialize security provider
            java.security.Security.addProvider(new org.bouncycastle.jce.provider.BouncyCastleProvider());

            // Create blockchain with difficulty 3
            Blockchain blockchain = new Blockchain(3);

            System.out.println("Mining genesis block...");
            System.out.println("Genesis Block Hash: " + blockchain.getLatestBlock().getHash());
            System.out.println();

            // Create wallets
            System.out.println("Creating wallets...");
            Wallet walletA = new Wallet();
            Wallet walletB = new Wallet();

            System.out.println("Wallet A address: " + walletA.getWalletAddress().substring(0, 30) + "...");
            System.out.println("Wallet B address: " + walletB.getWalletAddress().substring(0, 30) + "...");
            System.out.println();

            // ========== MINING BLOCK 1: Wallet B mines and gets reward ==========
            System.out.println("=== WALLET B STARTS MINING BLOCK 1 ===");

            Block block1 = new Block(blockchain.getLatestBlock().getHash());

            // Create coinbase transaction (mining reward for Wallet B)
            Transaction coinbase1 = Transaction.createCoinbase(walletB.getPublicKey(), Blockchain.getMiningReward());
            System.out.println("Coinbase transaction created: " + Blockchain.getMiningReward() + " coins for Wallet B (miner)");
            block1.addTransaction(coinbase1, blockchain);

            // Add a regular transaction from A to B (will fail - A has no balance yet)
            System.out.println("\nAttempting transaction from A to B for 100 coins...");
            Transaction tx1 = new Transaction(walletA.getPublicKey(), walletB.getPublicKey(), 100f);
            tx1.generateSignature(walletA.getPrivateKey());
            block1.addTransaction(tx1, blockchain);

            // Mine the block
            System.out.println("\nWallet B is mining... (finding hash with 3 leading zeros)");
            blockchain.addBlock(block1);
            blockchain.processTransactions(block1);

            System.out.println("Block 1 mined! Hash: " + block1.getHash());
            System.out.println();

            // Display balances after block 1
            System.out.println("=== Balances after Block 1 (Wallet B mined) ===");
            System.out.println("Wallet A balance: " + blockchain.getBalance(walletA.getWalletAddress()));
            System.out.println("Wallet B balance: " + blockchain.getBalance(walletB.getWalletAddress()) + " (mining reward!)");
            System.out.println();

            // ========== BLOCK 2: Wallet B mines again with A's transaction ==========
            System.out.println("=== WALLET B STARTS MINING BLOCK 2 ===");

            Block block2 = new Block(blockchain.getLatestBlock().getHash());

            // Coinbase for Wallet B (mining again)
            Transaction coinbase2 = Transaction.createCoinbase(walletB.getPublicKey(), Blockchain.getMiningReward());
            block2.addTransaction(coinbase2, blockchain);

            // Now Wallet A tries to send to Wallet B (will fail - still no balance)
            System.out.println("Attempting transaction from A to B for 50 coins...");
            Transaction tx2 = new Transaction(walletA.getPublicKey(), walletB.getPublicKey(), 50f);
            tx2.generateSignature(walletA.getPrivateKey());
            block2.addTransaction(tx2, blockchain);

            // Mine
            System.out.println("\nWallet B is mining...");
            blockchain.addBlock(block2);
            blockchain.processTransactions(block2);

            System.out.println("Block 2 mined! Hash: " + block2.getHash());
            System.out.println();

            // Display balances
            System.out.println("=== Balances after Block 2 ===");
            System.out.println("Wallet A balance: " + blockchain.getBalance(walletA.getWalletAddress()) + " (still 0 - no coins!)");
            System.out.println("Wallet B balance: " + blockchain.getBalance(walletB.getWalletAddress()) + " (2 × mining reward)");
            System.out.println();

            // ========== BLOCK 3: Wallet A mines and gets coins, then sends to B ==========
            System.out.println("=== WALLET A STARTS MINING BLOCK 3 ===");

            Block block3 = new Block(blockchain.getLatestBlock().getHash());

            // Coinbase for Wallet A (now A gets coins!)
            Transaction coinbase3 = Transaction.createCoinbase(walletA.getPublicKey(), Blockchain.getMiningReward());
            System.out.println("Coinbase transaction: " + Blockchain.getMiningReward() + " coins for Wallet A (miner)");
            block3.addTransaction(coinbase3, blockchain);

            // Now Wallet A can send to Wallet B!
            System.out.println("Transaction from A to B for 75 coins...");
            Transaction tx3 = new Transaction(walletA.getPublicKey(), walletB.getPublicKey(), 75f);
            tx3.generateSignature(walletA.getPrivateKey());
            block3.addTransaction(tx3, blockchain);

            // Mine
            System.out.println("\nWallet A is mining...");
            blockchain.addBlock(block3);
            blockchain.processTransactions(block3);

            System.out.println("Block 3 mined! Hash: " + block3.getHash());
            System.out.println();

            // Final balances
            System.out.println("=== Final Balances ===");
            System.out.println("Wallet A balance: " + blockchain.getBalance(walletA.getWalletAddress())
                    + " (100 mined - 75 sent = 25)");
            System.out.println("Wallet B balance: " + blockchain.getBalance(walletB.getWalletAddress())
                    + " (200 from mining + 75 received = 275)");
            System.out.println();

            // Verify blockchain integrity
            System.out.println("=== Blockchain Validation ===");
            System.out.println("Blockchain is valid: " + blockchain.isChainValid());
            System.out.println();

            // Display blockchain info
            System.out.println("=== Blockchain Info ===");
            System.out.println("Total blocks: " + blockchain.getChain().size());
            System.out.println("Mining difficulty: " + blockchain.getDifficulty());
            System.out.println("Mining reward: " + Blockchain.getMiningReward());
            System.out.println();

            // Show transactions in each block
            System.out.println("=== Block Transactions ===");
            for (int i = 0; i < blockchain.getChain().size(); i++) {
                Block b = blockchain.getChain().get(i);
                System.out.println("Block " + i + " (" + b.getTransactions().size() + " transactions):");
                for (Transaction t : b.getTransactions()) {
                    String sender = t.isCoinbase() ? "COINBASE" : "Wallet A";
                    String recipient = t.getRecipientAddress().substring(0, 10) + "...";
                    System.out.println("  → " + sender + " sends " + t.getAmount() + " to " + recipient);
                }
            }

            // Print visual blockchain state
            System.out.println();
            printBlockchainState(blockchain, walletA, walletB);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Prints a visual representation of the blockchain state
     */
    private static void printBlockchainState(Blockchain blockchain, Wallet walletA, Wallet walletB) {
        System.out.println();
        System.out.println("╔═══════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                    BLOCKCHAIN VISUAL STATE                                    ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════════════════════╝");
        System.out.println("Legend: [MINING] = Creates new coins  |  [TRANSFER] = Moves existing coins");
        System.out.println();

        for (int i = 0; i < blockchain.getChain().size(); i++) {
            Block block = blockchain.getChain().get(i);

            // Block header
            if (i == 0) {
                System.out.println("+---------- GENESIS BLOCK (Block #" + i + ") --------------------------------+");
            } else {
                System.out.println("+---------- BLOCK #" + i + " -----------------------------------------------------+");
            }

            // Block info
            String hashDisplay = block.getHash().length() > 64 ? block.getHash().substring(0, 64) + "..." : block.getHash();
            String prevDisplay = block.getPreviousHash().length() > 64 ? block.getPreviousHash().substring(0, 64) + "..." : block.getPreviousHash();

            System.out.println("| Hash:      " + hashDisplay);
            System.out.println("| Previous:  " + prevDisplay);
            System.out.println("| Timestamp: " + block.getTimestamp() + " | Nonce: " + block.getNonce() + " | Diff: " + blockchain.getDifficulty());

            // Transactions
            if (block.getTransactions().isEmpty()) {
                System.out.println("| Transactions: (none)");
            } else {
                System.out.println("| Transactions:");
                for (int j = 0; j < block.getTransactions().size(); j++) {
                    Transaction tx = block.getTransactions().get(j);

                    if (tx.isCoinbase()) {
                        String miner = walletB != null && tx.getRecipient().equals(walletB.getPublicKey()) ? "Wallet B" :
                                      (walletA != null && tx.getRecipient().equals(walletA.getPublicKey()) ? "Wallet A" : "Unknown");
                        System.out.println("|   [" + (j + 1) + "] [MINING]   " + miner + " gets " + tx.getAmount() + " coins");
                    } else {
                        String sender = walletA != null && tx.getSender().equals(walletA.getPublicKey()) ? "Wallet A" : "Unknown";
                        String recipient = walletB != null && tx.getRecipient().equals(walletB.getPublicKey()) ? "Wallet B" :
                                         (walletA != null && tx.getRecipient().equals(walletA.getPublicKey()) ? "Wallet A" : "Unknown");
                        System.out.println("|   [" + (j + 1) + "] [TRANSFER] " + sender + " -> " + tx.getAmount() + " -> " + recipient);
                    }
                }
            }

            System.out.println("+-------------------------------------------------------------------------+");

            // Arrow to next block
            if (i < blockchain.getChain().size() - 1) {
                System.out.println("                          |");
                System.out.println("                          V");
                System.out.println();
            }
        }

        // Wallet balances
        System.out.println();
        System.out.println("+-------------------------------------------------------------------------+");
        System.out.println("|                         WALLET BALANCES                                |");
        System.out.println("+-------------------------------------------------------------------------+");
        System.out.println("|  Wallet A: " + String.format("%8.1f", blockchain.getBalance(walletA.getWalletAddress())) + " coins  |  Wallet B: " + String.format("%8.1f", blockchain.getBalance(walletB.getWalletAddress())) + " coins  |");
        System.out.println("+-------------------------------------------------------------------------+");
    }
}
