//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[inline]
pub(super) fn levenshtein(a: &str, b: &str) -> u8 {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (alen, blen) = (a.len(), b.len());
    let mut dp = vec![vec![0u16; blen + 1]; alen + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i as u16;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j as u16;
    }
    for i in 1..=alen {
        for j in 1..=blen {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }
    dp[alen][blen].min(255) as u8
}
