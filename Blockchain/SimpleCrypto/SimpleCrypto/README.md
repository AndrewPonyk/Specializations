# SimpleCrypto

A lightweight, educational cryptocurrency implementation in Java demonstrating core blockchain concepts with minimal external dependencies.

## Project Overview

This project implements a simple blockchain-based cryptocurrency with the following features:

- **Proof of Work (PoW)** mining with adjustable difficulty
- **Coinbase transactions** - miners receive rewards for mining blocks
- **ECDSA digital signatures** for transaction security
- **Wallet management** with public/private key pairs
- **Merkle Tree** root calculation for transaction verification
- **Blockchain validation** to ensure chain integrity

## Project Structure

```
SimpleCrypto/
├── pom.xml                                          # Maven configuration
└── src/main/java/com/ap/
    ├── Block.java                                   # Block with mining & Merkle root
    ├── Blockchain.java                              # Chain management & validation
    ├── Main.java                                    # Demo application
    ├── StringUtil.java                              # SHA-256 hashing & ECDSA signatures
    ├── Transaction.java                             # Signed transactions & Coinbase
    └── Wallet.java                                  # ECDSA key pair management
```

## Code Description

### Block.java
Represents a single block in the blockchain.

| Component | Description |
|-----------|-------------|
| `hash` | SHA-256 hash of the block |
| `previousHash` | Hash of the previous block (chain linkage) |
| `timestamp` | Unix timestamp when block was created |
| `nonce` | Counter used in Proof of Work mining |
| `transactions` | List of transactions in this block |
| `merkleRoot` | Root hash of the Merkle tree of transactions |

**Key Methods:**
- `calculateHash()` - Computes SHA-256 hash of block data
- `mineBlock(int difficulty)` - Proof of Work mining to find hash with leading zeros
- `addTransaction()` - Validates and adds transaction to block

### Blockchain.java
Manages the entire blockchain.

| Component | Description |
|-----------|-------------|
| `chain` | Ordered list of blocks |
| `balances` | Wallet address to coin balance mapping |
| `difficulty` | Mining difficulty (number of leading zeros required) |
| `MINING_REWARD` | Coins awarded to miners per block (100) |

**Key Methods:**
- `createGenesisBlock()` - Creates the first block
- `addBlock()` - Adds a new mined block to chain
- `isChainValid()` - Validates integrity of entire blockchain
- `processTransactions()` - Updates balances after block is mined

**Note:** `addGenesisBalance()` exists for testing but demo uses real mining (coinbase) instead

### Transaction.java
Represents a funds transfer between wallets.

| Component | Description |
|-----------|-------------|
| `sender` | Public key of sender (null for coinbase) |
| `recipient` | Public key of recipient |
| `amount` | Amount to transfer |
| `signature` | ECDSA digital signature |
| `isCoinbase` | True if this is a mining reward transaction |

**Key Methods:**
- `calculateHash()` - Generates unique transaction ID
- `generateSignature()` - Signs transaction with private key
- `verifySignature()` - Verifies signature with public key
- `createCoinbase()` - Creates a mining reward transaction (sender = null)

### Wallet.java
Manages cryptographic keys and transactions.

| Component | Description |
|-----------|-------------|
| `privateKey` | ECDSA private key for signing |
| `publicKey` | ECDSA public key for verification |
| `transactions` | List of outgoing transactions |

**Key Methods:**
- `generateKeyPair()` - Creates 256-bit ECDSA key pair using BouncyCastle
- `getWalletAddress()` - Returns Base64 encoded public key

**Note:** `sendFunds()` exists but Main.java creates transactions directly for demonstration

### StringUtil.java
Utility class for cryptographic operations.

**Key Methods:**
- `applySha256()` - Computes SHA-256 hash
- `getStringFromKey()` - Converts keys to Base64 strings
- `applyECDSASig()` - Creates ECDSA signature
- `verifyECDSASig()` - Verifies ECDSA signature
- `getMerkleRoot()` - Computes Merkle tree root from transactions

### Main.java
Demonstration program showing the cryptocurrency in action with real mining rewards.

## Execution

### Prerequisites
- Java 11 or higher
- Maven 3.6+

### Running the Application

```bash
# Navigate to project directory
cd /path/to/SimpleCrypto

# Compile and run
mvn clean compile exec:java -Dexec.mainClass="com.ap.Main"
```

### What Happens During Execution

```
1. INITIALIZATION
   ├── BouncyCastle security provider loaded
   └── Blockchain created with difficulty = 3

2. GENESIS BLOCK
   ├── First block mined (hash starts with "000")
   └── Added to chain

3. WALLET CREATION
   ├── Wallet A created (new ECDSA key pair)
   ├── Wallet B created (new ECDSA key pair)
   └── Both start with 0 coins (must mine to earn!)

4. MINING DEMO - Real Mining with Coinbase Rewards
   ┌──────────────────────────────────────────────────────────────┐
   │                                                             │
   │   Start:  Wallet A = 0,  Wallet B = 0                       │
   │           ↓                                                 │
   │   Block 1: Wallet B mines → B = 100, A = 0                  │
   │           ↓                                                 │
   │   Block 2: Wallet B mines → B = 200, A = 0                  │
   │           ↓                                                 │
   │   Block 3: Wallet A mines → A = 100, B = 200                │
   │           ↓                                                 │
   │   Tx:     A sends 75 to B → A = 25, B = 275                 │
   │                                                             │
   └──────────────────────────────────────────────────────────────┘

   Key Concept: Coinbase Transactions
   ┌────────────────────────────────────────────────────────────┐
   │  • Each block starts with a coinbase transaction           │
   │  • sender = null (creates NEW coins!)                      │
   │  • recipient = miner's wallet address                      │
   │  • amount = MINING_REWARD (100 coins)                      │
   │  • No signature needed (mining is proof of work)           │
   └────────────────────────────────────────────────────────────┘

5. VALIDATION
   ├── Blockchain integrity verified
   └── All block hashes and links confirmed valid
```

### Sample Output

```
=== Simple Cryptocurrency Demo ===

Mining genesis block...
Genesis Block Hash: 000abc123...

Creating wallets...
Wallet A address: MFkwEwYHKoZIjvCAQyBUd...
Wallet B address: MFkwEwYHKoZIjvCBQyBUd...

=== WALLET B STARTS MINING BLOCK 1 ===
Coinbase transaction: 100.0 coins minted for miner

Attempting transaction from A to B for 100 coins...
Transaction failed: Not enough funds. Sender balance: 0.0, Amount: 100.0

Wallet B is mining... (finding hash with 3 leading zeros)
Block 1 mined! Hash: 000def456...

=== Balances after Block 1 (Wallet B mined) ===
Wallet A balance: 0.0
Wallet B balance: 100.0 (mining reward!)

=== WALLET B STARTS MINING BLOCK 2 ===
Coinbase transaction: 100.0 coins minted for miner

Attempting transaction from A to B for 50 coins...
Transaction failed: Not enough funds. Sender balance: 0.0, Amount: 50.0

Wallet B is mining...
Block 2 mined! Hash: 000ghi789...

=== Balances after Block 2 ===
Wallet A balance: 0.0 (still 0 - no coins!)
Wallet B balance: 200.0 (2 × mining reward)

=== WALLET A STARTS MINING BLOCK 3 ===
Coinbase transaction: 100.0 coins minted for miner

Transaction from A to B for 75 coins...
Transaction signature verified successfully

Wallet A is mining...
Block 3 mined! Hash: 000jkl012...

=== Final Balances ===
Wallet A balance: 25.0 (100 mined - 75 sent = 25)
Wallet B balance: 275.0 (200 from mining + 75 received = 275)

=== Blockchain Validation ===
Blockchain is valid: true

=== Blockchain Info ===
Total blocks: 4
Mining difficulty: 3
Mining reward: 100.0

=== Block Transactions ===
Block 0 (0 transactions):
Block 1 (1 transactions):
  → COINBASE sends 100.0 to MFkwEwYHKo...
Block 2 (1 transactions):
  → COINBASE sends 100.0 to MFkwEwYHKo...
Block 3 (2 transactions):
  → COINBASE sends 100.0 to MFkwEwYHKo...
  → Wallet A sends 75.0 to MFkwEwYHKo...
```

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| BouncyCastle | 1.70 | ECDSA signature provider |

## Security Considerations

⚠️ **This is an educational project, not production-ready!**

- No P2P network (single JVM instance)
- No mempool (transactions go directly to blocks)
- No persistent storage (in-memory only)
- Simplified consensus (single miner)
- No transaction fees
- No input validation on amounts

## Blockchain Vulnerabilities Analysis

| Vulnerability | Demonstrable | Reason |
|---------------|--------------|--------|
| 51% Attack | ✅ Yes | Can create private fork and replace chain |
| Sybil Attack | ⚠️ Partial | No real P2P network |
| Selfish Mining | ✅ Yes | Can demonstrate block withholding |
| Smart Contract Bugs | ❌ No | No smart contract VM |
| Reentrancy | ❌ No | No smart contracts |
| Front-running | ❌ No | No mempool |
| Private Key Compromise | ✅ Yes | Can sign with stolen key |
| Eclipse Attack | ❌ No | Single-node architecture |
| Dust Attack | ✅ Yes | Can send minimal amounts |
| Flash Loan | ❌ No | No DeFi protocols |

## Learning Outcomes

This project demonstrates:
1. How blocks are linked via cryptographic hashes
2. Proof of Work mining and difficulty adjustment
3. **Coinbase transactions** - how new coins are minted through mining
4. **Why you can't spend what you don't have** - transactions fail without sufficient balance
5. Public-key cryptography for transaction signing
6. Merkle trees for efficient transaction verification
7. Blockchain validation and integrity checking

## License

Educational use only.
