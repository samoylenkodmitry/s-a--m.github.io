---
layout: leetcode-entry
title: "3501. Maximize Active Section with Trade II"
permalink: "/leetcode/problem/2026-07-22-3501-maximize-active-section-with-trade-ii/"
leetcode_ui: true
entry_slug: "2026-07-22-3501-maximize-active-section-with-trade-ii"
---

[3501. Maximize Active Section with Trade II](https://leetcode.com/problems/maximize-active-section-with-trade-ii/solutions/8412830/kotlin-by-samoylenkodmitry-er7o/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22072026-3501-maximize-active-section?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8j-w662x-3g)

https://dmitrysamoylenko.com/leetcode/

![22.07.2026.webp](/assets/leetcode_daily_images/22.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1428

#### Problem TLDR

Queries l..r of max ones after replacing surrounding zeros in range

#### Intuition

Didn't solve.
```j
    // 10 minute: how to shrink the sliding window and update max?
    // 16 minute: no idea, lets look at hints - segment tree, no hope for that
```
Sort ranges by R. Scan string, track current zeros group and previous zeros group start and lengths.
BIT answers the question what is the max value in the suffix from L. We are filling the BIT by current position so it is always the range L..R.
Update BIT at start of the left group with value of both group sizes.
Query BIT by L of the query.

Corner case: BIT misses the first group if L happens to be at zeros. To overcome, track zeros counts and next zeros position for each position suffixes.

#### Approach

* threre are segment tree and sparse table solution which i didn't get yet.

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maxActiveSectionsAfterTrade(s: String, q: Array<IntArray>) = run {
        val n = s.length; val ones = s.count { it == '1' }; val bit = IntArray(n + 2)
        fun up(i: Int, v: Int) { var x = i + 1; while (x > 0) { bit[x] = max(bit[x], v); x -= x and -x } }
        fun qu(i: Int)=run{ var x = i + 1; var v = 0; while (x <= n + 1) { v = max(v, bit[x]); x += x and -x };v}
        val z = IntArray(n + 1); val nxt = IntArray(n + 1) { n }; val byR = q.indices.groupBy { q[it][1] }
        for (j in n - 1 downTo 0) if (s[j] == '0') { z[j] = 1 + z[j + 1]; nxt[j] = j } else nxt[j] = nxt[j + 1]
        val res = IntArray(q.size); var pStart = 0; var pLen = 0; var cLen = 0
        for (i in s.indices) {
            if (s[i] == '0') { cLen++; if (pLen > 0) up(pStart, pLen + cLen) }
            else if (cLen > 0) { pStart = i - cLen; pLen = cLen; cLen = 0 }
            byR[i]?.forEach { qi ->
                val l = q[qi][0]; val g2 = if (s[l] == '0') nxt[l + z[l]] else n
                val g = if (g2 <= i) minOf(z[l], i - l + 1) + minOf(z[g2], i - g2 + 1) else 0
                res[qi] = ones + maxOf(qu(l), g)
            }
        }; res
    }
```
```rust

```

