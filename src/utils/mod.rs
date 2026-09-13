pub mod file_utils;
pub mod project_structure_formatter;
pub mod threads;
pub mod token_estimator;
pub mod prompt_compressor;

/// String utilities for UTF-8 safe slicing.
///
/// Byte-indexed slicing of multi-byte content (Chinese, emoji, ...) panics
/// with `byte index is not a char boundary`. These helpers move the cut
/// point to the nearest valid boundary instead.

/// Largest byte index <= `max_bytes` that is a valid char boundary.
pub fn floor_char_boundary(s: &str, max_bytes: usize) -> usize {
    if s.len() <= max_bytes {
        return s.len();
    }
    let mut i = max_bytes;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Smallest byte index >= `min_bytes` that is a valid char boundary.
pub fn ceil_char_boundary(s: &str, min_bytes: usize) -> usize {
    if min_bytes >= s.len() {
        return s.len();
    }
    let mut i = min_bytes;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Truncate `s` to at most `max_bytes` without splitting a character.
pub fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    &s[..floor_char_boundary(s, max_bytes)]
}

/// Slice `s` starting from `from_bytes` without splitting a character.
/// `from_bytes` is moved forward to the next valid boundary if needed.
pub fn slice_from_char_boundary(s: &str, from_bytes: usize) -> &str {
    &s[ceil_char_boundary(s, from_bytes)..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_ascii_unchanged() {
        assert_eq!(truncate_at_char_boundary("abcdef", 3), "abc");
    }

    #[test]
    fn test_truncate_multibyte_floors_to_boundary() {
        // "中" is 3 bytes each; 4 bytes would split the 2nd character
        let s = "中文中文";
        assert_eq!(s.len(), 12);
        assert_eq!(truncate_at_char_boundary(s, 4), "中");
        assert_eq!(truncate_at_char_boundary(s, 12), "中文中文");
        assert_eq!(truncate_at_char_boundary(s, 100), "中文中文");
    }

    #[test]
    fn test_slice_from_multibyte_ceils_to_boundary() {
        // "中文中文": boundaries at 0,3,6,9,12 (3 bytes per char)
        let s = "中文中文";
        assert_eq!(slice_from_char_boundary(s, 4), "中文");
        assert_eq!(slice_from_char_boundary(s, 3), "文中文");
        assert_eq!(slice_from_char_boundary(s, 0), s);
        assert_eq!(slice_from_char_boundary(s, 12), "");
        assert_eq!(slice_from_char_boundary(s, 100), "");
    }

    #[test]
    fn test_truncate_never_panics_on_any_offset() {
        let s = "aé中😀b";
        for n in 0..=s.len() + 2 {
            let _ = truncate_at_char_boundary(s, n);
            let _ = slice_from_char_boundary(s, n);
        }
    }
}
