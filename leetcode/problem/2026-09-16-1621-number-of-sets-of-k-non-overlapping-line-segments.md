---
layout: leetcode-entry
title: "1621. Number of Sets of K Non-Overlapping Line Segments"
permalink: "/leetcode/problem/2026-09-16-1621-number-of-sets-of-k-non-overlapping-line-segments/"
leetcode_ui: true
entry_slug: "2026-09-16-1621-number-of-sets-of-k-non-overlapping-line-segments"
---

[1621. Number of Sets of K Non-Overlapping Line Segments](https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments/solutions/8524253/kotlin-rust-by-samoylenkodmitry-ypzn/) medium
[substack](https://dmitriisamoilenko.substack.com/p/16092026-1621-number-of-sets-of-k?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/qb9qlT-kyec)

https://dmitrysamoylenko.com/leetcode/

![16.09.2026.webp](/assets/leetcode_daily_images/16.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1484

#### Problem TLDR

Ways to peek K intervals from N points

#### Intuition

DFS DP: choose between continue, stop, start and stop-start
Math combinatorics: each interval is two points, meaning we are choosing 2k points all at once from n+k-1 total possible points nCr (n+k-1  2k)

#### Approach

* 1D dp solution is row-by-row Pascal's Triangle

#### Complexity

- Time complexity:
$$O(nk)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun numberOfSets(n: Int, k: Int): Int {
        var num = 1L; var den = 1L; val M = 1_000_000_007L
        for (i in 1..2 * k) { num = num * (n + k - i) % M; den = den * i % M }
        return (num * den.toBigInteger().modInverse(M.toBigInteger()).toLong() % M).toInt()
    }
```
```rust
    pub fn number_of_sets(n: i32, k: i32) -> i32 {
        let mut dp = vec![0; 2 * k as usize + 1]; dp[0] = 1;
        for _ in 1..n + k { for j in (1..dp.len()).rev() {
            dp[j] = (dp[j] + dp[j - 1]) % 1_000_000_007 } }
        dp[2 * k as usize]
    }
```

