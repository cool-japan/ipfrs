// @cool-japan/ipfrs — WebAssembly bindings for IPFRS
// This file re-exports from the wasm-pack generated dist/
let wasm;
try {
  wasm = require('./dist/ipfrs_wasm');
} catch (e) {
  wasm = {};
  console.warn('[ipfrs] WASM bundle not found. Run `npm run build` first.');
}

module.exports = wasm;
module.exports.VERSION = '0.3.0';
