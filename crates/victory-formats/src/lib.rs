//! Platform-level data formats, shared by every Xbox Series title.
//!
//! Each module is pure (bytes in, bytes out, no I/O): [`texture`] (the
//! GPU's tiled texture layouts and copies to and from linear rows). Planned:
//! BCPack textures, audio codecs (XMA2, Opus), and the zlib streams used by
//! hardware decompression.

pub mod bcpack;
mod detile;
pub mod texture;
pub mod xma;
