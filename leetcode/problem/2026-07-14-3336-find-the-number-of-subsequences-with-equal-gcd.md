---
layout: leetcode-entry
title: "3336. Find the Number of Subsequences With Equal GCD"
permalink: "/leetcode/problem/2026-07-14-3336-find-the-number-of-subsequences-with-equal-gcd/"
leetcode_ui: true
entry_slug: "2026-07-14-3336-find-the-number-of-subsequences-with-equal-gcd"
---

[3336. Find the Number of Subsequences With Equal GCD](https://leetcode.com/problems/find-the-number-of-subsequences-with-equal-gcd/solutions/8396354/kotlin-rust-by-samoylenkodmitry-sagu/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14072026-3336-find-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LnMMjFhXwfg)

https://dmitrysamoylenko.com/leetcode/

![14.07.2026.webp](/assets/leetcode_daily_images/14.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1420

#### Problem TLDR

Equal gcd subsequencies

#### Intuition

```j
    // 200 length and 200 max
    // the gcd can be fixed?
    // 6 minute: i have no idea, go for hints: dp[i][gcd1][gcd2]
    // 21 minute: wrong answer 615/622 test case
```
Used the hint.
* dp [i] [gcd1] [gcd2] take to the first or take to the second or skip

#### Approach

* the bottom up: flatten the gcd1xgcd2 table, use only the previous table, result is a diagonal sum

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
    fun subsequencePairCount(n: IntArray): Long {
        val dp = HashMap<Int, Long>(); val M = 1000000007L
        fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
        fun f(i: Int, x: Int, y: Int): Long = if (i == n.size)
            if (x == y) 1L else 0L else dp.getOrPut(i*40401+x*201+y) {
            (f(i+1, gcd(n[i],x), y) + f(i+1, x, gcd(n[i],y)) + f(i+1, x, y)) % M }
        return (f(0, 0, 0) - 1 + M) % M
    }
```
```rust
    pub fn subsequence_pair_count(n: Vec<i32>) -> i32 {
        let (mut dp, M) = ([0i64; 201*201], 1_000_000_007); dp[0] = 1;
        let g = |mut a: usize, mut b: usize| { while b > 0 {(a,b)=(b,a%b)} a };
        for v in n { let mut nxt = dp;
            for i in 0..201*201 { if dp[i] > 0 {
                let g1 = g(i/201, v as usize) * 201 + i%201;
                let g2 = i/201 * 201 + g(i%201, v as usize);
                nxt[g1] = (nxt[g1] + dp[i]) % M;
                nxt[g2] = (nxt[g2] + dp[i]) % M
            }} dp = nxt }
        ((1..201).map(|i| dp[i * 201 + i]).sum::<i64>() % M) as i32
    }
```

