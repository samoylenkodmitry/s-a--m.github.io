---
layout: leetcode-entry
title: "3405. Count the Number of Arrays with K Matching Adjacent Elements"
permalink: "/leetcode/problem/2025-06-17-3405-count-the-number-of-arrays-with-k-matching-adjacent-elements/"
leetcode_ui: true
entry_slug: "2025-06-17-3405-count-the-number-of-arrays-with-k-matching-adjacent-elements"
---

[3405. Count the Number of Arrays with K Matching Adjacent Elements](https://leetcode.com/problems/count-the-number-of-arrays-with-k-matching-adjacent-elements/description/) hard
[blog post](https://leetcode.com/problems/count-the-number-of-arrays-with-k-matching-adjacent-elements/solutions/6853008/kotlin-rust-by-samoylenkodmitry-ssue/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17062025-3405-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/oelVEDn2chQ)
![1.webp](/assets/leetcode_daily_images/76eb20a3.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1022

#### Problem TLDR

Combinations k equal siblings 1..m in [n] array #hard #combinatorics #math

#### Intuition

Didn't solve.
Some thougths:

```j
    // n = 4, m = 2, k = 2
    // [1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2], [2, 1, 1, 1], [2, 2, 1, 1] [2, 2, 2, 1]
    // dp[i] = (1..m).sum { a  (1..m).sum { b  (a == b) + dp[i + 2] } }
    // 10^5 x 10^5 will give TLE/MLE
    // combinatorics? all perimutations excluding the banned
    // 1 1 2  1 2 2  2 1 1  2 2 1 k = 1
    // 1 2 1         2 1 2        k = 0  ban
    // 1 1 1         2 2 2        k = 2  ban
    //                            k max = n - 1
    // (0..kMax) = all perm "111".toString(2)
    // stil don't know how to count perm for each `k`
    // hints are pointing to the DP, but how it is not TLE?
    // (26 minute, give up, its combinatorics)
    // m * C(n - 1, k) * (m - 1) ** (n - 1 - k)

```

If I understood it right:
* stars and bars: the bars are equal parts, we have `k` of them on `n-1` positions: `C(n-1,k)`
* first is `1..m`
* the stars can be `1..m - 1(prev)` at `(n-1) - k` positions (after bars are placed): `(m-1)^(n-1-k)`

#### Approach

* this time I recognized my inability to solve combinatorics much faster than 1 hour (26 minute gave up)
* memoize `a^b % m`: it is derived from math `a ^ b = (a^2)^(b/2) * a^(b%2), b /= 2, a = a * a`
* memoize combinations nCr (n choose r): `n!/(n-r)!r!` = `(1..n)/(1..r)(1..n-r) = (1..n)/(1..n-r) / (1..r) =  (n-r+1)..n/1..r`
* memoize how to calculate this with `% M`: `Fermat` theorem `x^(m-1)=1 %m  is eq   x^(m-2)=x^-1  %m`, so `1/1..r % m  is eq  (1..r)^m-2 % m` or `den^-1 % m = den^(m-2) %m` https://en.wikipedia.org/wiki/Fermat%27s_little_theorem
* of course it is impossible for me to solve without some huge investment into combinatorics theory, but some tricks are worth to learn

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 28ms
    fun countGoodArrays(n: Int, m: Int, k: Int): Int {
        val M = 1_000_000_007; val m = 1L * m; var nCr = 1L; var den = 1L
        fun pow(a: Long, b: Int): Long =
            if (b == 0) 1L else (pow((a * a) % M, b / 2) * if (b % 2 > 0) a else 1) % M
        for (i in 1..k) { nCr = (nCr * (n - i)) % M; den = (den * i) % M }
        nCr = (nCr * pow(den, M - 2)) % M
        return ((((m * nCr) % M) * pow(m - 1, n - k - 1)) % M).toInt()
    }

```
```rust

// 28ms
    pub fn count_good_arrays(n: i32, m: i32, k: i32) -> i32 {
        let (M, n, m, k, mut nCr, mut den) = (1_000_000_007, n as i64, m as i64, k as i64, 1, 1);
        fn pow(a: i64, b: i64, M: i64) -> i64 {
            if b == 0 { 1 } else { (pow((a * a) % M, b / 2, M) * if (b % 2 > 0) { a } else { 1 }) % M }
        }
        for i in 1..=k { nCr = (nCr * (n - i)) % M; den = (den * i) % M }
        nCr = (nCr * pow(den, M - 2, M)) % M;
        ((((m * nCr) % M) * pow(m - 1, n - k - 1, M)) % M) as i32
    }

```
```c++

// 25ms
#define M 1000000007
int countGoodArrays(int n, int m, int k) {
    auto p = [](long a, long b) {
        long r = 1;
        while (b) { if (b & 1) r = r * a % M; a = a * a % M; b >>= 1; }
        return r;
    };
    long x = 1, y = 1;
    for (int i = 1; i <= k; ++i) { x = x * (n - i) % M; y = y * i % M; }
    return x * p(y, M - 2) % M * m % M * p(m - 1, n - k - 1) % M;
}

```

