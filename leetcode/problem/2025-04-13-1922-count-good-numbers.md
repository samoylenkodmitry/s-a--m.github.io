---
layout: leetcode-entry
title: "1922. Count Good Numbers"
permalink: "/leetcode/problem/2025-04-13-1922-count-good-numbers/"
leetcode_ui: true
entry_slug: "2025-04-13-1922-count-good-numbers"
---

[1922. Count Good Numbers](https://leetcode.com/problems/count-good-numbers/description/) medium
[blog post](https://leetcode.com/problems/count-good-numbers/solutions/6645719/kotlin-rust-by-samoylenkodmitry-7yfc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13042025-1922-count-good-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3KOGEU37_9g)
![1.webp](/assets/leetcode_daily_images/6b09b8bc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/957

#### Problem TLDR

Count `(0,2,4,6,8)(2,3,5,7)` generated strings of length n #medium #math

#### Intuition

Looking at the pattern:

```j

    // 0 1 2 3 4 5 6 ... n - 1
    // 0
    // 2 2
    // 4 3
    // 6 5
    // 8 7
    // 1 -> 5
    // 2 -> 5*4
    // 3 -> 5^2 *4
    // 4 -> 5^2 * 4^2
    // ..
    // n -> 5^((n+1)/2) * 4^(n/2)

    // "big exp % mod" pattern
    // x^y = x^(2*y/2) = (x^2) ^ (y/2) * (x^2 ^ (y%2))

```

The problem is about how to compute `5^((n+1)/2) * 4^(n/2)`.
We can do this with a math trick: `x^y = x^(2*y/2) = (x^2) ^ (y/2) * (x^2 ^ (y%2))`

#### Approach

* BigInteger also works

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countGoodNumbers(n: Long): Int {
        val M = 1_000_000_007; var x = 20L; var e = n / 2; var r = 1 + 4 * (n % 2)
        while (e > 0) { if (e % 2 > 0) r = (r * x) % M; x = (x * x) % M; e /= 2 }
        return r.toInt()
    }

```
```kotlin

    fun countGoodNumbers(n: Long): Int {
        val M = 1_000_000_007L
        fun p(x: Long, e: Long): Long =
            if (x < 2 || e < 1L) 1 else if (e < 2L) x
            else (p((x * x) % M, e / 2) * p(x, e % 2)) % M
        return (p(5L, (n + 1) / 2) * p(4L, n / 2) % M).toInt()
    }

```
```kotlin

    fun countGoodNumbers(n: Long): Int {
        val (a, e, m) = listOf(20L, n / 2, 1_000_000_007L).map { it.toBigInteger() }
        return a.modPow(e, m).multiply((1L + 4L * (n % 2)).toBigInteger()).mod(m).intValueExact()
    }

```
```rust

    pub fn count_good_numbers(n: i64) -> i32 {
        let (M, mut x, mut e, mut r) = (1_000_000_007, 20, n / 2, 1 + 4 * (n & 1));
        while e > 0 { if e & 1 > 0 { r = (r * x) % M }; x = (x * x) % M; e >>= 1 }
        return r as _
    }

```
```c++

    int countGoodNumbers(long long n) {
        long long M = 1e9 + 7, x = 20, e = n / 2, r = 1 + 4 * (n & 1);
        while (e) { if (e & 1) r = (r * x) % M; x = (x * x) % M; e >>= 1;}
        return (int) r;
    }

```

