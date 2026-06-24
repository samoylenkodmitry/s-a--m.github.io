---
layout: leetcode-entry
title: "3700. Number of ZigZag Arrays II"
permalink: "/leetcode/problem/2026-06-24-3700-number-of-zigzag-arrays-ii/"
leetcode_ui: true
entry_slug: "2026-06-24-3700-number-of-zigzag-arrays-ii"
---

[3700. Number of ZigZag Arrays II](https://leetcode.com/problems/number-of-zigzag-arrays-ii/solutions/8355439/kotlin-by-samoylenkodmitry-3m0u/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24062026-3700-number-of-zigzag-arrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/y33AHlHLF3E)

https://dmitrysamoylenko.com/leetcode/

![24.06.2026.webp](/assets/leetcode_daily_images/24.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1400

#### Problem TLDR

Ways to make l..r zigzag sequence

#### Intuition

Didn't solve.
Make a transition matrix that depends only on the prvious step: up-up/down-down are disabled, up-down and down-up are enabled
(look youtube video).
The dp[l..r goes up, l..r goes down] = [1 1 1 1  1 1.. 1 1 1]is the number of ways.
Multiplay by transition matrix n times to get total number of ways for each cell+direction.
Sum all the counts.

#### Approach

* the trick here is to encode the directions in a one big transition matrix [u-u u-d / d-u d-d]

#### Complexity

- Time complexity:
$$O(rlogn)$$

- Space complexity:
$$O(r)$$

#### Code

```kotlin
    fun zigZagArrays(sz: Int, l: Int, r: Int): Int {
        val M = 1000000007; val m = r - l + 1; val S = 2 * m
        var b = LongArray(S * S); var dp = LongArray(S) { 1L }; var p = sz - 1L
        for (i in 0..<m) for (j in 0..<m)
            if (j < i) b[i * S + j + m] = 1L else if (j > i) b[(i + m) * S + j] = 1L
        while (p > 0) {
            if (p % 2 == 1L) dp = LongArray(S).also { nDp ->
                for (i in 0..<S) if (dp[i] > 0) for (j in 0..<S)
                    nDp[j] = (nDp[j] + dp[i] * b[i * S + j]) % M
            }
            b = LongArray(S * S).also { nB ->
                for (i in 0..<S) for (k in 0..<S) if (b[i * S + k] > 0) for (j in 0..<S)
                    nB[i * S + j] = (nB[i * S + j] + b[i * S + k] * b[k * S + j]) % M
            }
            p /= 2
        }
        return (dp.sum() % M).toInt()
    }
```
```rust

```

