# **IPFRS (Inter-Planet File RUST System) (構想案 v0.3.0)**

**Version:** 0.3.0 (Unified Strategy: "The Fast & The Wise")

**Architect:** TensorLogic Architect

**Date:** 2026-01-18

**Based on:** v0.1.0 (Rust Rewrite) & v0.2.0 (Neural-Symbolic)

## **1\. Executive Summary: "From Pipes to Synapses"**

**IPFRS (Inter-Planet File RUST System)** は、世界のWeb3インフラを「GoからRustへ」と刷新する構造改革プロジェクトであると同時に、次世代AI "TensorLogic" のための「惑星規模の分散頭脳」を実現する物理基盤である。

本バージョン（v0.3.0）は、v0.1.0の\*\*「圧倒的な省メモリ・高スループット性能」と、v0.2.0の「推論と学習の分散パイプライン」を単一のアーキテクチャに統合する。  
従来のIPFSが「静的なファイルの倉庫」であったのに対し、IPFRSは「思考する高速道路（Thinking Highway）」\*\*として機能する。Armエッジデバイスからデータセンターまで、ゼロコピーでテンソルを循環させ、地球規模の推論（Planetary Inference）をリアルタイムで実行可能にする。

## **2\. The "R" Philosophy: Unified Core Values**

IPFRSのアイデンティティである "R" は、インフラ（Infrastructure）とインテリジェンス（Intelligence）の両面を包含する4つの柱に進化する。

### **Infrastructure Layer (from v0.1.0)**

* **Rust (Memory Safety & Zero-GC):**  
  * Go言語のGCによるレイテンシを排除し、ミリ秒単位の応答性が求められるリアルタイムAI推論を支える。  
  * Tokio非同期ランタイムによる高密度な並行処理を実現。  
* **Robust (Arm-Optimized):**  
  * AWS GravitonからRaspberry Pi、NVIDIA Jetsonまで、Armアーキテクチャ（NEON/SVE）にネイティブ最適化。  
  * エッジでの電力対性能比（Performance per Watt）を最大化。

### **Intelligence Layer (from v0.2.0)**

* **Reasoning-Ready (Logic-Aware):**  
  * 単なるBlob（塊）ではなく、TensorLogicの「論理項」と「計算グラフ」を理解し、分散バックワード・チェイニング（推論連鎖）をサポート。  
* **Resilient (Vector Semantics):**  
  * CID（完全一致）による検索に加え、分散ベクトルインデックス（HNSW on DHT）による「意味検索」を統合。

## **3\. High-Level Architecture: The Bi-Layer Stack**

Kuboのレガシーな設計を刷新し、TensorLogicランタイムとメモリ空間を共有する「バイレイヤー（二層）」構造を採用する。

graph TD  
    subgraph "Application Space (TensorLogic Runtime)"  
        AI\_Agent\[AI Agent / LLM\]  
        Inference\[Inference Engine\<br\>(Reasoning / Learning)\]  
    end

    subgraph "IPFRS Unified Node (v0.3.0)"  
        direction TB  
          
        subgraph "Logical Layer (The Brain)"  
            Semantic\[\<b\>Semantic Router\</b\>\<br\>(Vector Search / Logic Solver)\]  
            DiffStore\[\<b\>Differentiable Storage\</b\>\<br\>(Gradient Tracking / Git-for-Tensors)\]  
        end

        FFI\[\<b\>Zero-Copy Interface (Apache Arrow)\</b\>\<br\>Shared Memory Boundary\]

        subgraph "Physical Layer (The Body)"  
            Exchange\[\<b\>TensorSwap\</b\>\<br\>(Bitswap \+ GraphSync Optimized)\]  
            BlockStore\[\<b\>Rust Native Store\</b\>\<br\>(Sled / ParityDB)\]  
            Net\[\<b\>Network Stack\</b\>\<br\>(libp2p / QUIC / WebTransport)\]  
        end  
    end

    AI\_Agent \--\> Inference  
    Inference \<--\>|Direct Memory Access| FFI  
    Semantic \--\> FFI  
    Semantic \--\> Exchange  
    Exchange \<--\> Net  
    Net \<--\> Internet((\<b\>COOLJAPAN\<br\>Knowledge Mesh\</b\>))

### **3.1 Unified Component Stack: Deep Dive**

v0.1.0の「Rust実装の詳細」とv0.2.0の「TensorLogic連携の役割」を融合させた、各コンポーネントの技術仕様。

#### **Layer 1: The Interface (Zero-Copy Boundary)**

* **Technology:** Apache Arrow / Safetensors / Rust FFI  
* **Role:**  
  * **Memory Sharing:** IPFRSがネットワークから受信したデータブロック（ページ）を、コピーすることなくそのままTensorLogicランタイムのメモリ空間としてマッピングする。  
  * **Serialization-Free:** 従来のJSON/Protobuf変換のオーバーヘッドを排除。推論エンジンは、IPFRSのキャッシュを「自分のメモリ」として直接読み書きする。

#### **Layer 2: The Logical Core (Semantic Router)**

* **Technology:** HNSW (Hierarchical Navigable Small World) / DiskANN  
* **Role:**  
  * **Dual-Resolution:** 「正確なハッシュ値（CID）」による従来の検索と、「意味ベクトル（Embedding）」による類似検索をハイブリッドで解決する。  
  * **Logic Solver:** 「AならばB」といった推論ルールに対し、その証明に必要なデータ（Fact）がネットワーク上のどこにあるかを特定するルーティングエンジン。

#### **Layer 3: The Transport (TensorSwap Protocol)**

* **Technology:** QUIC (quinn) / Bitswap (Custom) / GraphSync  
* **Role:**  
  * **Tensor Streaming:** 巨大なLLMの重みやテンソルデータを、パケットロスに強いQUICストリーム上で高速転送する。  
  * **Dependency Awareness:** 単なるブロック要求ではなく、計算グラフ（Einsum Graph）の依存関係に基づき、「推論に必要な順序」でデータを優先的にフェッチする。

#### **Layer 4: The Storage (Differentiable Blockstore)**

* **Technology:** Sled (Embedded DB) / ParityDB / IPLD  
* **Role:**  
  * **Version Control:** 学習データの変更履歴（勾配更新）をGitのように管理。過去の任意の時点のモデル状態を瞬時に復元可能。  
  * **Hot/Cold Tiering:** 頻繁にアクセスされる「短期記憶」はSled（メモリ/SSD）に、長期的な「知識」はParquet形式などで圧縮保存する。

#### **Layer 5: The Network (The Neural Mesh)**

* **Technology:** rust-libp2p / Kademlia DHT  
* **Role:**  
  * **Edge Optimized:** 省メモリ設計により、Armベースのエッジデバイス（Raspberry Pi, Jetson）でもフルノードとして参加可能。  
  * **Semantic DHT:** 従来のKademlia DHTを拡張し、ベクトル空間上の近傍探索をプロトコルレベルでサポートする。

### **3.2 The Workflow: How It Thinks**

IPFRSにおけるデータの流れは、ファイルの送受信ではなく「思考のプロセス」として設計されている。

1. **Thinking (Query):**  
   * AIエージェントが「論理クエリ（例: knows(user, ?concept)）」を発行。  
   * Semantic Routerがローカル知識を確認し、不足分をネットワークへ問い合わせる。  
2. **Recalling (Discovery):**  
   * DHTがクエリベクトルに近い知識を持つノード群（Experts）を特定。  
   * 意味的な類似性に基づいて、最適なデータソースを選択。  
3. **Synapsing (Transport):**  
   * TensorSwapが接続を確立し、計算グラフの実行に必要な順序でテンソルをストリーミング開始。  
   * QUICにより、モバイル環境などの不安定な回線でも低遅延を維持。  
4. **Reasoning (Execution):**  
   * 受信したデータはZero-Copy Interface経由で即座に推論エンジンのメモリ空間に出現。  
   * 推論結果が確定次第、新たな「知識」としてIPFRSにキャッシュ・永続化される。

## **4\. Key Capabilities & Differentiators**

### **4.1 Zero-Copy Tensor Transport (性能×知能)**

* **課題:** 従来のPython \+ IPFS構成では、\[Goのメモリ\] → \[ソケット\] → \[Pythonのメモリ\] → \[GPUメモリ\] と多重のコピーが発生していた。  
* **解決:** IPFRSはRustで書かれており、TensorLogicランタイム（Rust製）と同じプロセス内で動作可能。ネットワークから受信したパケット（QUICフレーム）を、そのままApache Arrow配列としてマッピングし、**ゼロコピーで推論エンジンに引き渡す**。  
* **結果:** モデルロード時間をKubo比で1/10以下に短縮。

### **4.2 Distributed Backward Chaining (分散推論)**

* **動作:** ノードAが knows(X, Y) を推論する際、知識が不足していれば、IPFRSがDHTを通じて「述語 knows の定義」や「関連事実」を持つノードB, Cを特定。  
* **自律性:** 必要なデータだけをオンデマンドで取得（TensorSwap）し、ノードAのローカルメモリ上で推論を完結させる。

### **4.3 Differentiable Storage (微分可能なストレージ)**

* **学習の民主化:** 世界中に分散したデータで学習を行う際、各データに対する「勾配（Gradient）」の更新履歴をIPLD（Merkle DAG）として管理。  
* **Provenance:** 「どのデータによってモデルがどう変化したか」を追跡可能にし、XAI（説明可能なAI）の物理的な証跡とする。

## **5\. Consolidated Roadmap (3-Month "Genesis" Plan)**

インフラ構築（v0.1）とAI統合（v0.2）を並行させ、最短で「思考するネットワーク」を立ち上げる。

### **Phase 1: The Foundation (Month 1\) \- "Connecting the Dots"**

* **Objective:** Rustによるlibp2pノードの確立と、TensorLogic型定義の統合。  
* **Tasks:**  
  1. rust-libp2p ベースのノード立ち上げ（QUIC優先）。  
  2. tensorlogic::ir::Term をIPLDへシリアライズするコーデックの実装。  
  3. 基本的な ipfs add/cat 互換コマンドの実装（CLI）。  
* **Milestone:** TensorLogicのデータ構造をIPFRS経由で保存・取得できるCLIツール。

### **Phase 2: The Synapse (Month 2\) \- "Streaming the Thought"**

* **Objective:** ゼロコピー転送の実装と、TensorSwapプロトコル。  
* **Tasks:**  
  1. **TensorSwap:** Safetensors形式のデータをストリーミング転送する独自プロトコル実装。  
  2. **Apache Arrow Binding:** 受信バッファをTensorLogicランタイムに直結するFFI層。  
  3. Arm/Neon命令セットを使用したハッシング/暗号化の最適化。  
* **Milestone:** Raspberry Pi上のIPFRSノードが、サーバーからLLMの一部を高速ロードして推論を実行するデモ。

### **Phase 3: The Awakening (Month 3\) \- "Planetary Reasoning"**

* **Objective:** 意味検索と分散推論の実証。  
* **Tasks:**  
  1. **Semantic Router:** DHTにHNSWインデックスを統合（プロトタイプ）。  
  2. **Gradient Tracking:** 学習結果（勾配）のIPLDへの書き込みテスト。  
  3. COOLJAPANエコシステム上での大規模負荷テスト。  
* **Milestone:** 複数のノードが協調して一つの論理パズルを解く「分散推論」の成功。

## **6\. Conclusion: The Infrastructure of Understanding**

IPFRS v0.3.0は、単なるストレージの再発明ではない。それは、人類の知識（データ）と、機械の知能（推論）を\*\*「同じ物理法則（Protocol）」\*\*の下で統一する試みである。

Rustによる堅牢な実装（The Body）と、TensorLogicによる柔軟な推論（The Brain）が融合することで、IPFRSはCOOLJAPANエコシステムを支える\*\*「自律分散型知識メッシュ」\*\*の中核となる。

**Project Status:** Unified & Ready for Sprint 1

**Command:** cargo new ipfrs \--bin \--edition 2024