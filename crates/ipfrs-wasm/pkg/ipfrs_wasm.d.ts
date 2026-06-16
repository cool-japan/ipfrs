/* tslint:disable */
/* eslint-disable */
/**
 * Initialise the IPFRS WASM module.
 */
export function start(): void;

/**
 * Compute a CIDv1 (base32-lower, SHA2-256, raw codec) string for the given bytes.
 */
export function compute_cid(data: Uint8Array): string;

/**
 * Return the ipfrs-wasm version string.
 */
export function version(): string;

/**
 * Verify that `data` matches `cid`.
 */
export function verify_cid(cid: string, data: Uint8Array): boolean;

/**
 * Add bytes using a temporary in-memory client and return the CID.
 */
export function add_bytes(data: Uint8Array): Promise<string>;

/**
 * Retrieve bytes from an IpfrsClient by CID.
 */
export function get_bytes(client: IpfrsClient, cid: string): Promise<Uint8Array | undefined>;

/**
 * IPFRS in-memory client for WebAssembly (ephemeral, HashMap-backed).
 */
export class IpfrsClient {
  free(): void;
  constructor();
  add(data: Uint8Array): Promise<string>;
  get(cid: string): Promise<Uint8Array>;
  has(cid: string): boolean;
  listCids(): string[];
  stats(): string;
  delete(cid: string): boolean;
}

/**
 * IPFRS persistent client backed by IndexedDB (browser only).
 */
export class IpfrsClientPersistent {
  free(): void;
  constructor(db_name: string);
  add(data: Uint8Array): Promise<string>;
  get(cid: string): Promise<Uint8Array | undefined>;
  has(cid: string): Promise<boolean>;
  delete(cid: string): Promise<boolean>;
  count(): Promise<number>;
}

/**
 * IndexedDB-backed content-addressed block store.
 */
export class IndexedDbStore {
  free(): void;
  static open(db_name: string): Promise<IndexedDbStore>;
  put(data: Uint8Array): Promise<string>;
  get(cid: string): Promise<Uint8Array | undefined>;
  has(cid: string): Promise<boolean>;
  delete(cid: string): Promise<boolean>;
  count(): Promise<number>;
}

// ---------------------------------------------------------------------------
// WasmBlockStore — in-memory IndexedDB-compatible block store
// ---------------------------------------------------------------------------

/**
 * In-memory block store that mirrors the IndexedDB async interface.
 *
 * In production browser builds this would delegate to `idb-keyval` JS glue
 * for real persistence.  The current implementation keeps all blocks in
 * Rust heap memory; data is lost on page refresh.
 *
 * Keys are CID strings; values are raw byte payloads.
 * Inserting the same CID twice overwrites the previous value (last-write-wins).
 *
 * @example
 * ```typescript
 * const store = new WasmBlockStore();
 * store.put("bafkreia…", new TextEncoder().encode("hello"));
 * const bytes = store.get("bafkreia…"); // Uint8Array | undefined
 * console.log(store.len(), store.total_bytes());
 * store.clear();
 * ```
 */
export class WasmBlockStore {
  free(): void;
  /** Create a new, empty WasmBlockStore. */
  constructor();
  /**
   * Insert or overwrite the block identified by `cid` with `data`.
   * If a block with the same CID already exists it is replaced (last-write-wins).
   */
  put(cid: string, data: Uint8Array): void;
  /**
   * Retrieve the raw bytes stored under `cid`.
   * Returns `undefined` when the CID is absent.
   */
  get(cid: string): Uint8Array | undefined;
  /** Return `true` if `cid` is present in the store. */
  has(cid: string): boolean;
  /**
   * Remove the block identified by `cid`.
   * Returns `true` if the block existed and was removed, `false` otherwise.
   */
  delete(cid: string): boolean;
  /** Return the number of blocks currently in the store. */
  len(): number;
  /** Return `true` if the store contains no blocks. */
  is_empty(): boolean;
  /** Return all stored CID strings in sorted order. */
  keys(): string[];
  /** Return the total size (in bytes) of all stored payloads. */
  total_bytes(): number;
  /** Remove all blocks from the store. */
  clear(): void;
}

// ---------------------------------------------------------------------------
// WebRTC peer connectivity
// ---------------------------------------------------------------------------

/**
 * Browser-to-browser peer connection handle — caller side.
 *
 * Creates an SDP offer, accepts an SDP answer, exchanges ICE candidates,
 * then sends content-addressed blocks over a WebRTC data channel.
 *
 * @example
 * ```typescript
 * const peer = new IpfrsPeer("peer-a");
 * const offerSdp = await peer.create_offer();
 * // … forward offerSdp to the answerer via your signalling channel …
 * const answerSdp: string = /* received from answerer *\/;
 * await peer.set_answer(answerSdp);
 * // … exchange ICE candidates …
 * // Once connected:
 * peer.send_block("bafk…", new Uint8Array([1, 2, 3]));
 * ```
 */
export class IpfrsPeer {
  free(): void;
  /**
   * Create a new caller-side peer connection.
   * @param peer_id - Application-level identifier for this peer.
   */
  constructor(peer_id: string);
  /**
   * Generate an SDP offer and set it as the local description.
   * @returns The offer SDP string to forward to the answerer.
   */
  create_offer(): Promise<string>;
  /**
   * Apply the SDP answer received from the remote (answerer) peer.
   * @param answer_sdp - The answer SDP string received from the answerer.
   */
  set_answer(answer_sdp: string): Promise<void>;
  /**
   * Add an ICE candidate received from the remote peer.
   * @param candidate_json - JSON-serialised IceCandidate object.
   */
  add_ice_candidate(candidate_json: string): void;
  /**
   * Send a content-addressed block to the remote peer over the data channel.
   *
   * Wire format: [4-byte BE cid_len][cid bytes][block data bytes]
   *
   * The data channel must be open (is_connected() === true).
   *
   * @param cid - CIDv1 string for the block.
   * @param data - Raw block bytes.
   */
  send_block(cid: string, data: Uint8Array): void;
  /**
   * Return the application-level peer identifier.
   */
  peer_id(): string;
  /**
   * Return true when the data channel is open and ready to transmit blocks.
   */
  is_connected(): boolean;
}

/**
 * Browser-to-browser peer connection handle — answerer side.
 *
 * Receives an SDP offer from the caller, creates an SDP answer, and
 * exchanges ICE candidates.
 *
 * @example
 * ```typescript
 * const offerSdp: string = /* received from caller *\/;
 * const answerer = await IpfrsPeerAnswerer.from_offer(offerSdp);
 * const answerSdp = await answerer.create_answer();
 * // … send answerSdp back to the caller via your signalling channel …
 * // … exchange ICE candidates …
 * ```
 */
export class IpfrsPeerAnswerer {
  free(): void;
  /**
   * Create an answerer peer connection from the caller's SDP offer.
   * @param offer_sdp - The SDP offer string received from the caller.
   */
  static from_offer(offer_sdp: string): Promise<IpfrsPeerAnswerer>;
  /**
   * Generate an SDP answer and set it as the local description.
   * @returns The answer SDP string to forward to the caller.
   */
  create_answer(): Promise<string>;
  /**
   * Add an ICE candidate received from the remote (caller) peer.
   * @param candidate_json - JSON-serialised IceCandidate object.
   */
  add_ice_candidate(candidate_json: string): Promise<void>;
  /**
   * Return true when the ICE connection has reached a usable state.
   */
  is_connected(): boolean;
}
