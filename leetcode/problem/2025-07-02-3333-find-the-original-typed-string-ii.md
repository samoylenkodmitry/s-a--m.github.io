---
layout: leetcode-entry
title: "3333. Find the Original Typed String II"
permalink: "/leetcode/problem/2025-07-02-3333-find-the-original-typed-string-ii/"
leetcode_ui: true
entry_slug: "2025-07-02-3333-find-the-original-typed-string-ii"
---

[3333. Find the Original Typed String II](https://leetcode.com/problems/find-the-original-typed-string-ii/description/) hard
[blog post](https://leetcode.com/problems/find-the-original-typed-string-ii/solutions/6910352/kotlin-rust-by-samoylenkodmitry-mkvq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/2072025-3333-find-the-original-typed?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Fsd9yYRBgxU)
![1.webp](/assets/leetcode_daily_images/facf5102.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1037

#### Problem TLDR

ways to remove duplicates, preserve length at least k #hard #dp

#### Intuition

Didn't solved.

```j

    // aaabbb   k=3   len=6
    // aaabbb
    //
    // aaabb   b=3, 3-1=2, 6-2=4
    // aaab

    // aabbb   a:2, 6-2=4
    // abbb
    //
    // aabb  combinatorics? a_variants * b_variants - bad_comb
    // aab
    // abb
    //
    // bad:
    // ab
    // dp? it is 10^5 * 2000 will give TLE, use hint(14 minute)
    // at most k - 1
    // bad combination length is at most k - 1

    // ab - single bad
    //      if k=5
    // ab, abb, abbb, aab, aabb, aaab - bad
    // how to count them?
    // aaabbbcccbbb    k=7
    // a  b  c  b      min=4, can take +2 on each or +1 +1 on any pair
    // aa
    // aaa
    // aa bb
    // aa    cc
    // aa       bb
    //    bb
    //    bbb
    //(aa bb)
    //    bb cc
    //    bb    bb
    //       cc
    //       ccc
    //(aa    cc)
    //   (bb cc)
    //       cc bb
    //          bb
    //          bbb
    //(aa       bb)
    //   (bb    bb)
    //      (cc bb)     (15 bad), total=1*3a*3b*3c*3b = 9*9=81
    //                  ans = 81-15 = 66

    // choose up to 2 from aa|bb|cc|bb = choose 1 + choose 2
    // C(1, 4) + C(2, 4) ??? how to choose from: a|bbbb|cc

    // choose (k - 1 - min) from islands  a|bbb|cc|b|aa
    //                           only non-singles
    //                                    bb|c|a
    // 43 minute, look for solution

```

* the main hardness is how to choose `bad` variants from a non-equal buckets (after we remove all the minimal required chars)
* `n` - size of minimum non-repeating buckets
* `g` - groups, with filtered out minimal required values `aa` becomes `a`, `b` becomes `` and filtered out; we only interested in the `repeating_count - 1` values to choose from
* use `DP[curr_bucket] = sum_{kk-g(i)}(DP[prev_bucket]) = PS[curr]-PS[kk-g(i)]`

#### Approach

* good solution from /u/votrubac/ https://leetcode.com/problems/find-the-original-typed-string-ii/solutions/5982440/optimized-tabulation/

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 416ms
    fun possibleStringCount(w: String, k: Int): Int {
        val M = 1_000_000_007L; val g = ArrayList<Int>()
        var c = 1; var all = 1L; var n = k
        for (i in 1..w.length) if (i < w.length && w[i] == w[i - 1]) ++c
            else { --n; if (c - 1 > 0) g += c - 1; all = (all * c) % M; c = 1 }
        if (n <= 0) return all.toInt()
        val dp = Array(2) { LongArray(n) }; dp[0][0] = 1L; val ps = LongArray(n + 1)
        for (i in 0..<g.size) for (kk in 0..<n) {
            ps[kk + 1] = (ps[kk] + dp[i % 2][kk]) % M
            dp[1 - (i % 2)][kk] = (ps[kk + 1] - ps[max(0, kk - g[i])]) % M
        }
        var bad = 0L; for (i in 0..<n) bad = (bad + dp[g.size % 2][i]) % M
        return ((all + M - bad) % M).toInt()
    }

```
```rust

// 49ms
    pub fn possible_string_count(w: String, k: i32) -> i32 {
        let g = w.as_bytes().chunk_by(|a, b| a == b).map(|c| c.len() as i64).collect::<Vec<_>>();
        let M = 1_000_000_007; let mut all = 1; for &c in &g { all = (all * c) % M };
        if k as usize <= g.len() { return all as i32 }; let n = k as usize - g.len();
        let g = g.iter().filter(|&&c| c > 1).map(|&c| c - 1).collect::<Vec<_>>();
        let mut dp = vec![vec![0; n]; 2]; dp[0][0] = 1i64; let mut ps = vec![0; n + 1];
        for i in 0..g.len() { for kk in 0..n {
            ps[kk + 1] = (ps[kk] + dp[i % 2][kk]) % M;
            dp[1 - (i % 2)][kk] = (ps[kk + 1] - ps[(0.max(kk as i64 - g[i])) as usize]) % M;
        }}
        let mut bad = 0; for i in 0..n { bad = (bad + dp[g.len() % 2][i]) % M }
        ((all + M - bad) % M) as i32
    }

```

#  1.07.2025
[3330. Find the Original Typed String I](https://leetcode.com/problems/find-the-original-typed-string-i/description/) easy
[blog post](https://leetcode.com/problems/find-the-original-typed-string-i/solutions/6905864/kotlin-rust-by-samoylenkodmitry-2tvw/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30062025-594-longest-harmonious-subsequence-65d?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/N7cO7hOoK3w)
![1.webp](/assets/leetcode_daily_images/085214b8.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1036

#### Problem TLDR

ways to remove duplicates #easy #counting

#### Intuition

Count duplicates, answer is sum of `count - 1`.
Corner case: duplicates must be adjacent.

#### Approach

* count `same chars islands`
* or just count equal adjacent pairs

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 136ms
    fun possibleStringCount(w: String) = 1 +
        w.windowed(2).count { it[0] == it[1] }

```
```kotlin

// 108ms
    fun possibleStringCount(w: String): Int {
        var cnt = 1; var p = '.'
        for (c in w) if (c == p) ++cnt else p = c
        return cnt
    }

```
```kotlin

// 98ms
    fun possibleStringCount(w: String): Int {
        var cnt = 1; var r = 0; var p = '.'
        for (c in w) if (c == p) ++r else { cnt += r; r = 0; p = c  }
        return cnt + r
    }

```
```rust

// 2ms
    pub fn possible_string_count(w: String) -> i32 {
       w.as_bytes().chunk_by(|a, b| a == b).collect::<Vec<_>>()
       .iter().map(|w| 0.max(w.len() as i32 - 1)).sum::<i32>() + 1
    }

```
```rust

// 0ms
    pub fn possible_string_count(w: String) -> i32 {
       1 + w.as_bytes().windows(2).filter(|w| w[0] == w[1]).count() as i32
    }

```
```c++

// 0ms
    int possibleStringCount(string w) {
        int cnt = 1; char p = '.';
        for (char c: w) c == p ? ++cnt : p = c;
        return cnt;
    }

```

