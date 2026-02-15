package com.ap;

import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.SecureRandom;

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

            System.out.println("Wallet A address: " + walletA.getWalletAddress().substring(0, 20) + "...");
            System.out.println("Wallet B address: " + walletB.getWalletAddress().substring(0, 20) + "...");
            System.out.println();

            // Give Wallet A some initial coins (like mining reward)
            blockchain.addGenesisBalance(walletA.getWalletAddress(), 500f);
            System.out.println("Wallet A initial balance (after mining): " + blockchain.getBalance(walletA.getWalletAddress()));
            System.out.println();

            // Create a transaction from Wallet A to Wallet B
            System.out.println("Creating transaction from A to B for 100 coins...");
            PublicKey pubKeyA = walletA.getPublicKey();
            PublicKey pubKeyB = walletB.getPublicKey();

            Transaction tx1 = new Transaction(pubKeyA, pubKeyB, 100f);
            tx1.generateSignature(walletA.getPrivateKey());

            if (tx1.verifySignature()) {
                System.out.println("Transaction signature verified!");
            }

            // Create a new block and add the transaction
            Block block1 = new Block(blockchain.getLatestBlock().getHash());
            block1.addTransaction(tx1, blockchain);

            System.out.println("\nMining block 1...");
            blockchain.addBlock(block1);
            blockchain.processTransactions(block1);

            System.out.println("Block 1 Hash: " + block1.getHash());
            System.out.println();

            // Display balances
            System.out.println("=== Balances after transaction ===");
            System.out.println("Wallet A balance: " + blockchain.getBalance(walletA.getWalletAddress()));
            System.out.println("Wallet B balance: " + blockchain.getBalance(walletB.getWalletAddress()));
            System.out.println();

            // Create another transaction
            System.out.println("Creating transaction from A to B for 50 coins...");
            Transaction tx2 = new Transaction(pubKeyA, pubKeyB, 50f);
            tx2.generateSignature(walletA.getPrivateKey());

            Block block2 = new Block(blockchain.getLatestBlock().getHash());
            block2.addTransaction(tx2, blockchain);

            System.out.println("\nMining block 2...");
            blockchain.addBlock(block2);
            blockchain.processTransactions(block2);

            System.out.println("Block 2 Hash: " + block2.getHash());
            System.out.println();

            // Final balances
            System.out.println("=== Final Balances ===");
            System.out.println("Wallet A balance: " + blockchain.getBalance(walletA.getWalletAddress()));
            System.out.println("Wallet B balance: " + blockchain.getBalance(walletB.getWalletAddress()));
            System.out.println();

            // Verify blockchain integrity
            System.out.println("=== Blockchain Validation ===");
            System.out.println("Blockchain is valid: " + blockchain.isChainValid());
            System.out.println();

            // Display blockchain info
            System.out.println("=== Blockchain Info ===");
            System.out.println("Total blocks: " + blockchain.getChain().size());
            System.out.println("Mining difficulty: " + blockchain.getDifficulty());

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
