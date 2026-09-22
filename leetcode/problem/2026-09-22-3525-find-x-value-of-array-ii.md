---
layout: leetcode-entry
title: "3525. Find X Value of Array II"
permalink: "/leetcode/problem/2026-09-22-3525-find-x-value-of-array-ii/"
leetcode_ui: true
entry_slug: "2026-09-22-3525-find-x-value-of-array-ii"
---

[3525. Find X Value of Array II](https://leetcode.com/problems/find-x-value-of-array-ii/solutions/8534430/kotlin-by-samoylenkodmitry-xgqf/) hard
[substack](https://dmitriisamoilenko.substack.com/p/22092026-3525-find-x-value-of-array?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UWfFrekrWsk)

https://dmitrysamoylenko.com/leetcode/

![22.09.2026.webp](/assets/leetcode_daily_images/22.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1490

#### Problem TLDR

Queries of count subarrays product%k = x for x in 0..<k in suffixes s..end

#### Intuition

```j
    // let's give up straigh from the start
    // know your limits
```
Each segment tree node stores counts[product%k] and the total product for the range L..node.
Queries do query segment tree s..end and result is counts[x] of the merged query result.
To merge ranges together A[..]B[..] we taking all counts of A and we additionally taking all counts of B[i] by continuing the A[i]*i%k.

#### Approach

* iterative segment tree: size 2*count, left 2*i, right 2*i+1, up i/2, query L%2>0, R%2>0 - goes horizontally+merges

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun resultArray(n: IntArray, k: Int, q: Array<IntArray>) = run {
        val c = 2 * n.size.takeHighestOneBit()
        fun e() = IntArray(k + 1).apply { this[k] = 1 }
        fun l(v: Int) = e().apply { this[k] = v % k; this[v % k] = 1 }
        fun m(a: IntArray, b: IntArray) = a.clone().apply {
            this[k] = a[k] * b[k] % k; for (i in 0..<k) this[a[k] * i % k] += b[i]
        }
        val t = Array(2 * c) { e() }; for (i in n.indices) t[c + i] = l(n[i])
        for (i in c - 1 downTo 1) t[i] = m(t[2 * i], t[2 * i + 1])
        q.map { (i, v, s, x) -> t[c + i] = l(v)
            var p = (c + i) / 2; while (p > 0) { t[p] = m(t[2 * p], t[2 * p + 1]); p /= 2 }
            var L = c + s; var R = 2 * c; var lr = e()
            while (L < R) { if (L % 2 > 0) lr = m(lr, t[L++]); L /= 2; R /= 2 }
            lr[x]
        }
    }
```
```rust

```

