---
layout: leetcode-entry
title: "3461. Check If Digits Are Equal in String After Operations I"
permalink: "/leetcode/problem/2025-10-23-3461-check-if-digits-are-equal-in-string-after-operations-i/"
leetcode_ui: true
entry_slug: "2025-10-23-3461-check-if-digits-are-equal-in-string-after-operations-i"
---

[3461. Check If Digits Are Equal in String After Operations I](https://leetcode.com/problems/check-if-digits-are-equal-in-string-after-operations-i/description/) easy
[blog post](https://leetcode.com/problems/check-if-digits-are-equal-in-string-after-operations-i/solutions/7294439/kotlin-rust-by-samoylenkodmitry-k9on/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23102025-3461-check-if-digits-are?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/igKy5PgHV14)

![3706fdee-bab0-4c11-8a4b-bc83632f6439 (1).webp](/assets/leetcode_daily_images/f06995f5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1151

#### Problem TLDR

Fold each pair to (a+b)%10 until 2 left #easy

#### Intuition

Just do the simulation

#### Approach

* use windows
* this problem has a hard solution for O(n): each digit repeats in Pascal triangle pattern coefficients, that is `nCr(size-2, i) % 10`. Then there are hard tricks to find %10.

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 86ms
    fun hasSameDigits(s: String): Boolean = if (s.length < 3) s[0] == s[1] else
        hasSameDigits(s.map {it-'0'}.windowed(2).map {it.sum()%10}.joinToString(""))

```
```kotlin

    fun hasSameDigits(s: String): Boolean {
        fun nCr(n: Int, k: Int): Int {
            if (k < 0 || k > n) return 0
            var k = if (k > n - k) n - k else k; var c = 1.toBigInteger()
            for (j in 1..k) c = c.multiply((n - k + j).toBigInteger())
                                 .divide(j.toBigInteger())
            return c.mod(10.toBigInteger()).toInt()
        }
        var a = 0; var b = 0
        for (i in s.indices) {
            a = (a + nCr(s.length - 2, i) * (s[i] - '0')) % 10
            b = (b + nCr(s.length - 2, i - 1) * (s[i] - '0')) % 10
        }
        return a == b
    }

```
```kotlin

    fun hasSameDigits(s: String): Boolean {
        fun pascal(p: Int) = Array(p) { IntArray(p) }.also {
            for (i in 0..<p) {
                it[i][0] = 1; it[i][i] = 1
                for (j in 1..<i) it[i][j] = (it[i-1][j-1] + it[i-1][j])%p
            }
        }
        val pas2 = pascal(2); val pas5 = pascal(5)
        fun nCrLucas(nn: Int, kk: Int, p: Int, m: Array<IntArray>): Int {
            var n = nn; var k = kk; var res = 1
            while (n > 0 || k > 0) {
                val ni = n % p; val ki = k % p; if (ki > ni) return 0
                res = (res * m[ni][ki]) % p
                n /= p; k /= p
            }
            return res
        }
        fun nCr10(n: Int, k: Int): Int {
            if (k < 0 || k > n) return 0
            val r2 = nCrLucas(n, k, 2, pas2); val r5 = nCrLucas(n, k, 5, pas5)
            var r = r5 + if ((r5 and 1) != (r2 and 1)) 5 else 0
            return r % 10
        }
        var a = 0; var b = 0
        for (i in s.indices) {
            a = (a + nCr10(s.length-2, i)   * (s[i] - '0')) % 10
            b = (b + nCr10(s.length-2, i-1) * (s[i] - '0')) % 10
        }
        return a == b
    }

```
```rust

// 0ms
    pub fn has_same_digits(s: String) -> bool {
        let mut v: Vec<_> = s.bytes().map(|c| c - b'0').collect();
        while v.len() >= 3 { v = v.windows(2).map(|w| (w[0] + w[1]) % 10).collect() }
        v[0] == v[1]
    }

```

