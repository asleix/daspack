//! Common runtime knobs for DASPack.

use std::sync::OnceLock;

/// Parse DASPACK_THREADS as a positive integer.
/// Returns Some(n) if n >= 1, otherwise None.
/// Any non-integer value is ignored (treated as unset).
pub fn env_threads() -> Option<usize> {
    match std::env::var("DASPACK_THREADS") {
        Ok(val) => {
            let s = val.trim();
            match s.parse::<usize>() {
                Ok(n) if n >= 1 => Some(n),
                _ => {
                    eprintln!(
                        "DASPack: ignoring invalid DASPACK_THREADS='{}' (must be a positive integer).",
                        val
                    );
                    None
                }
            }
        }
        Err(_) => None,
    }
}

/// Decide the effective thread count given an explicit argument.
/// Precedence:
///   1) explicit > 0
///   2) DASPACK_THREADS env (positive integer)
///   3) fallback to 1 (serial)
pub fn effective_threads(explicit: usize) -> usize {
    if explicit > 0 {
        explicit
    } else {
        env_threads().unwrap_or(1)
    }
}

/// Ensure a Rayon *global* thread-pool exists with `n` threads.
/// This runs only once; subsequent calls are no-ops.
/// If the global pool was already initialized elsewhere, we log and continue.
///
/// NOTE: call this *before* any Rayon usage to have the size take effect.
pub fn ensure_global_rayon_pool(n: usize) {
    static INIT_ONCE: OnceLock<()> = OnceLock::new();
    INIT_ONCE.get_or_init(|| {
        let n = n.max(1);
        if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(n).build_global() {
            // Most likely: "The global thread pool has already been initialized."
            eprintln!("DASPack: global Rayon pool init skipped ({e}).");
        } else if n > 1 {
            eprintln!("DASPack: initialized global Rayon pool with {n} thread(s).");
        }
    });
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_effective_threads() {
//         unsafe {
//             // Unset path → 1
//             std::env::remove_var("DASPACK_THREADS");
//             assert_eq!(effective_threads(0), 1);
//             // Explicit wins
//             assert_eq!(effective_threads(8), 8);
//             // Env wins when explicit==0
//             std::env::set_var("DASPACK_THREADS", "4");
//             assert_eq!(effective_threads(0), 4);
//             // Invalid env → fallback 1
//             std::env::set_var("DASPACK_THREADS", "bogus");
//             assert_eq!(effective_threads(0), 1);
//             std::env::set_var("DASPACK_THREADS", "0");
//             assert_eq!(effective_threads(0), 1);
//         }

//     }
// }
