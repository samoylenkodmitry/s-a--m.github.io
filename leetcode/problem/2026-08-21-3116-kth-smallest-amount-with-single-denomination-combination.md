---
layout: leetcode-entry
title: "3116. Kth Smallest Amount With Single Denomination Combination"
permalink: "/leetcode/problem/2026-08-21-3116-kth-smallest-amount-with-single-denomination-combination/"
leetcode_ui: true
entry_slug: "2026-08-21-3116-kth-smallest-amount-with-single-denomination-combination"
---

[3116. Kth Smallest Amount With Single Denomination Combination](https://leetcode.com/problems/kth-smallest-amount-with-single-denomination-combination/solutions/8473933/kotlin-by-samoylenkodmitry-2nxx/) hard
[substack](https://dmitriisamoilenko.substack.com/p/21082026-3116-kth-smallest-amount?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vMsh1NRYEEI)

https://dmitrysamoylenko.com/leetcode/

![21.08.2026.webp](/assets/leetcode_daily_images/21.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1458

#### Problem TLDR

Kth number in multipliers sequence of coins

#### Intuition

Didn't solve.
```j
    // 8 12 24
    // 8 16 24
    // 12   24
    //      24
    // how to remove duplicates?
    // i do not see how to apply incl-excl principle
    //
```
To remove duplicates remove the LCM sequences.

#### Approach

* the recursive ways is more natural

#### Complexity

- Time complexity:
$$O(log(k)2^n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun findKthSmallest(c: IntArray, k: Int): Long {
        fun gcd(a: Long, b: Long): Long = if (b == 0L) a else gcd(b, a % b)
        fun count(m: Long, i: Int = 0, L: Long = 1): Long = if (i==c.size) 0
        else (c[i] * L / gcd(L, 1L * c[i]))
            .let { nL -> count(m, i + 1, L) + m / nL - count(m, i + 1, nL) }
        var lo = 1L; var hi = 1L * c.min() * k
        while (lo <= hi) {
            val m = (lo + hi) / 2
            if (count(m) < k) lo = m + 1 else hi = m - 1
        }
        return lo
    }
```
```rust

```

