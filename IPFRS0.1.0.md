# **IPFRS: Inter-Planet File RUST System 構想案**

Version: 0.1.0 (Pre-Alpha Strategy)  
Architect: TensorLogic Architect  
Date: 2026-01-18

## **1\. Executive Summary**

**IPFRS** (Inter-Planet File RUST System) は、既存のIPFSリファレンス実装（Kubo/Go）が抱える構造的なパフォーマンス限界とリソース消費の問題を、Rust言語による完全再構築によって解決する次世代の分散ストレージ・ミドルウェアである。

本プロジェクトは、単なる「Rustへの移植」ではない。IPFSのプロトコル仕様（Wire Protocol）には完全に準拠しつつ、内部アーキテクチャを最新の非同期ランタイム（Tokio）とゼロコピー設計で刷新することで、\*\*「Edge環境における圧倒的な省メモリ性」**と**「データセンターレベルの高スループット」\*\*を両立させる。

これは、COOLJAPANエコシステムの基盤となるだけでなく、世界のWeb3インフラを「GoからRustへ」と塗り替える戦略的プロジェクト（Game Changer）である。

## **2\. The "R" Philosophy**

IPFRSのアイデンティティは **"R"** に集約される。

1. **R**ust (Memory Safety & No GC)  
   * Go言語のStop-the-world GCによるレイテンシ・スパイクを排除。  
   * 予測可能なメモリ使用量により、組込み機器（IoT/Edge AI）での安定稼働を保証。  
2. **R**obust (Arm-Optimized)  
   * Armアーキテクチャ（特にNEON/SVE命令）への最適化を施し、電力対性能比（Performance per Watt）を最大化。  
3. **R**apid (High Throughput)  
   * Irohで実証されたQUIC/HTTP3技術を取り入れつつ、レガシーIPFS網との互換性を維持。  
4. **R**evolution (Infrastructure Renewal)  
   * 肥大化したKuboコードベースを廃棄し、必要最小限かつ高密度なコードベースで再定義する。

## **3\. High-Level Architecture**

Kuboのモノリシックな設計を解体し、疎結合なレイヤー構造とする。

graph TD  
    User\[User / App / TensorLogic\] \--\>|HTTP / gRPC / FFI| API\[IPFRS API Layer\]  
      
    subgraph IPFRS\_Core \[IPFRS Core Engine\]  
        API \--\> Resolver\[Content Resolver\]  
        Resolver \--\> BlockStore\[Block Storage (Sled/RocksDB)\]  
        Resolver \--\> Exchange\[Bitswap / GraphSync\]  
          
        Exchange \--\> Network\[Network Stack (libp2p)\]  
        Network \--\> DHT\[DHT (Kademlia)\]  
        Network \--\> QUIC\[Transport (QUIC/TCP/WS)\]  
    end  
      
    QUIC \<--\> Internet((Global IPFS Swarm))

### **3.1 Component Stack**

| Layer | Component | Technology Selection | Strategy |
| :---- | :---- | :---- | :---- |
| **Interface** | API / Gateway | **Axum / Hyper** | 世界最速クラスのRust Webフレームワークを採用。Kubo互換APIと独自High-Speed APIを提供。 |
| **Logic** | Core Logic | **Custom Rust** | 非同期タスク制御。TensorLogic連携を見据えたFFI境界の最適化。 |
| **Storage** | Blockstore | **Sled / ParityDb** | GoのLevelDB/BadgerDBより高速な、RustネイティブKVSを採用。プラグイン可能にする。 |
| **Protocol** | Data Exchange | **Bitswap** (Custom) | rust-libp2pのプリミティブを使い、スループット重視で独自実装。 |
| **Network** | Networking | **rust-libp2p** | 既存の堅牢なエコシステムを利用。Tokioランタイム上でQUIC (quinn) を優先利用する設定。 |

## **4\. Key Differentiators vs Kubo**

なぜIPFRSなのか。Kubo（Go版）に対する明確な優位性を定義する。

| Feature | Kubo (Go) | IPFRS (Rust) | Impact |
| :---- | :---- | :---- | :---- |
| **GC Pause** | あり (数ms〜数百ms) | **なし (RAII)** | リアルタイムAI推論時のデータ供給に遅延を生まない。 |
| **Memory** | GC用に余分なヒープが必要 | **Minimal** | Armエッジデバイス(2GB RAM等)でもフルノード稼働が可能。 |
| **Embbed** | 困難 (プロセス間通信のみ) | **容易 (FFI)** | Python/Node.js/C++アプリ内にライブラリとして内蔵可能。 |
| **Architecture** | 2015年基準の設計 | **最新基準 (Async)** | 最新の並行処理パラダイムによるコア数スケーリング。 |

## **5\. Roadmap (3-Month "Blitzkrieg" Plan)**

750万行のシステムを構築可能なエンジニアリングリソースを前提とした、短期集中ロードマップ。

### **Phase 1: Foundation (Month 1\)**

* **目標:** 世界のIPFS網への接続（Hello World to the Galaxy）。  
* **実装:**  
  * rust-libp2p を用いたノード立ち上げ。  
  * PeerID生成、DHTへの参加（Kademlia）。  
  * CID (Multihash) の生成と検証ロジック。  
  * **Deliverable:** パブリックDHTからピアを見つけられるCLIツール。

### **Phase 2: Core Engine (Month 1.5 \- 2\)**

* **目標:** データの送受信と保存（The Heartbeat）。  
* **実装:**  
  * **Bitswapプロトコル**の実装（これが最難関かつ差別化ポイント）。  
  * ローカルBlockstore (Sled) の実装。  
  * ipfs add (Import) と ipfs cat (Export) の基本動作。  
* **Deliverable:** KuboノードからファイルをダウンロードできるIPFRSバイナリ。

### **Phase 3: Interface & Optimization (Month 3\)**

* **目標:** 実用性と性能証明（The Proof）。  
* **実装:**  
  * **HTTP Gateway (Axum)** の実装。  
  * Kubo互換 RPC API (v0) の一部実装。  
  * ベンチマークテストとプロファイリング調整。  
* **Deliverable:** Kubo比でメモリ使用量1/5、スループット3倍を実証するベンチマークレポート。

## **6\. Strategic Fit for "COOLJAPAN" & Arm**

### **6.1 TensorLogicとの連携**

IPFRSは、TensorLogic（AIエンジン）にとっての「分散共有メモリ」として機能する。

* **Model Distribution:** 巨大なLLMの重みデータを、中央サーバーなしでエッジデバイス群に高速配布。  
* **Data Provenance:** CIDによる改ざん不可能な学習データ管理。

### **6.2 Arm/SoftBankへのピッチ要素**

* **"IPFRS runs best on Arm"**  
  * AWS Graviton, Raspberry Pi, NVIDIA Jetsonなど、Arm系プロセッサでの動作を第一級市民（First-Class Citizen）とする。  
  * これをもって、データセンターからエッジまでを貫通する「統一データプレーン」として提案可能。

## **7\. Conclusion**

IPFRSは、既存のIPFSプロトコルの「再発明」ではなく、Rustという現代の武器を用いた\*\*「構造改革」\*\*である。  
20万行（推定）のコンパクトかつ高密度なコードベースは、旧来のKuboが抱える負債を一掃し、Web3とAI時代の新たなデファクト・スタンダードとなるポテンシャルを秘めている。  
Project Status: Ready to Launch  
Initial Command: cargo new ipfrs