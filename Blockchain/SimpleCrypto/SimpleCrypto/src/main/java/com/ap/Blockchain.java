package com.ap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Blockchain {
    private List<Block> chain;
    private Map<String, Float> balances;
    private int difficulty;
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

    public boolean isChainValid() {
        for (int i = 1; i < chain.size(); i++) {
            Block currentBlock = chain.get(i);
            Block previousBlock = chain.get(i - 1);

            if (!currentBlock.getHash().equals(currentBlock.calculateHash())) {
                return false;
            }

            if (!currentBlock.getPreviousHash().equals(previousBlock.getHash())) {
                return false;
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
