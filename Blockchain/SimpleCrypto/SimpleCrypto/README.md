# SimpleCrypto

A lightweight, educational cryptocurrency implementation in Java demonstrating core blockchain concepts with minimal external dependencies.

## Project Overview

This project implements a simple blockchain-based cryptocurrency with the following features:

- **Proof of Work (PoW)** mining with adjustable difficulty
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
    ├── Transaction.java                             # Signed transactions
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

**Key Methods:**
- `createGenesisBlock()` - Creates the first block
- `addBlock()` - Adds a new mined block to chain
- `isChainValid()` - Validates integrity of entire blockchain
- `processTransactions()` - Updates balances after block is mined

### Transaction.java
Represents a funds transfer between wallets.

| Component | Description |
|-----------|-------------|
| `sender` | Public key of sender |
| `recipient` | Public key of recipient |
| `amount` | Amount to transfer |
| `signature` | ECDSA digital signature |

**Key Methods:**
- `calculateHash()` - Generates unique transaction ID
- `generateSignature()` - Signs transaction with private key
- `verifySignature()` - Verifies signature with public key

### Wallet.java
Manages cryptographic keys and transactions.

| Component | Description |
|-----------|-------------|
| `privateKey` | ECDSA private key for signing |
| `publicKey` | ECDSA public key for verification |
| `transactions` | List of outgoing transactions |

**Key Methods:**
- `generateKeyPair()` - Creates 256-bit ECDSA key pair using BouncyCastle
- `sendFunds()` - Creates and signs a new transaction
- `getWalletAddress()` - Returns Base64 encoded public key

### StringUtil.java
Utility class for cryptographic operations.

**Key Methods:**
- `applySha256()` - Computes SHA-256 hash
- `getStringFromKey()` - Converts keys to Base64 strings
- `applyECDSASig()` - Creates ECDSA signature
- `verifyECDSASig()` - Verifies ECDSA signature
- `getMerkleRoot()` - Computes Merkle tree root from transactions

### Main.java
Demonstration program showing the cryptocurrency in action.

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
   └── Wallet A funded with 500 coins (genesis balance)

4. FIRST TRANSACTION (A → B: 100 coins)
   ├── Transaction created
   ├── Signed with Wallet A's private key
   ├── Signature verified
   ├── New block created
   ├── Block mined (finds hash with leading zeros)
   └── Balances updated: A=400, B=100

5. SECOND TRANSACTION (A → B: 50 coins)
   ├── Transaction created
   ├── Signed with Wallet A's private key
   ├── New block created and mined
   └── Balances updated: A=350, B=150

6. VALIDATION
   ├── Blockchain integrity verified
   └── All block hashes and links confirmed valid
```

### Sample Output

```
=== Simple Cryptocurrency Demo ===

Mining genesis block...
Genesis Block Hash: 000abc123...

Creating wallets...
Wallet A address: MFkwEwYHKoZIjvCAQ...
Wallet B address: MFkwEwYHKoZIjvCBQ...

Wallet A initial balance (after mining): 500.0

Creating transaction from A to B for 100 coins...
Transaction signature verified!

Mining block 1...
Block 1 Hash: 000def456...

=== Balances after transaction ===
Wallet A balance: 400.0
Wallet B balance: 100.0

Creating transaction from A to B for 50 coins...

Mining block 2...
Block 2 Hash: 000ghi789...

=== Final Balances ===
Wallet A balance: 350.0
Wallet B balance: 150.0

=== Blockchain Validation ===
Blockchain is valid: true

=== Blockchain Info ===
Total blocks: 3
Mining difficulty: 3
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
3. Public-key cryptography for transaction signing
4. Merkle trees for efficient transaction verification
5. Blockchain validation and integrity checking

## License

Educational use only.
