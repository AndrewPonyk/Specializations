package com.ap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// |su:5 BLOCKCHAIN: The ledger. A list of blocks linked by hash, making data tamper-evident.
public class Blockchain {
    private List<Block> chain;  // The actual chain of blocks
    private Map<String, Float> balances;  // |su:6 STATE: Track wallet balances (in production, compute from UTXOs)
    private int difficulty;  // Mining difficulty - how many leading zeros required in hash
    private static final float MINING_REWARD = 100f;

    public Blockchain(int difficulty) {
        this.chain = new ArrayList<>();
        this.balances = new HashMap<>();
        this.difficulty = difficulty;
        createGenesisBlock();
    }

    private void createGenesisBlock() {
        Block genesis = new Block("0");
        genesis.mineBlock(difficulty);
        chain.add(genesis);
    }

    public Block getLatestBlock() {
        return chain.get(chain.size() - 1);
    }

    public void addBlock(Block newBlock) {
        newBlock.mineBlock(difficulty);
        chain.add(newBlock);
    }

    // |su:7 INTEGRITY: Verify chain hasn't been tampered with. Checks: hash correctness & block linkage
    public boolean isChainValid() {
        for (int i = 1; i < chain.size(); i++) {
            Block currentBlock = chain.get(i);
            Block previousBlock = chain.get(i - 1);

            if (!currentBlock.getHash().equals(currentBlock.calculateHash())) {
                return false;  // Block data was modified
            }

            if (!currentBlock.getPreviousHash().equals(previousBlock.getHash())) {
                return false;  // Chain link broken - block was removed/inserted
            }
        }
        return true;
    }

    public void processTransactions(Block block) {
        for (Transaction tx : block.getTransactions()) {
            if (tx.getSender() != null) {
                balances.put(tx.getSenderAddress(), getBalance(tx.getSenderAddress()) - tx.getAmount());
            }
            balances.put(tx.getRecipientAddress(), getBalance(tx.getRecipientAddress()) + tx.getAmount());
        }
    }

    public float getBalance(String address) {
        return balances.getOrDefault(address, 0f);
    }

    public void addGenesisBalance(String address, float amount) {
        balances.put(address, getBalance(address) + amount);
    }

    public List<Block> getChain() {
        return chain;
    }

    public int getDifficulty() {
        return difficulty;
    }

    public static float getMiningReward() {
        return MINING_REWARD;
    }
}
