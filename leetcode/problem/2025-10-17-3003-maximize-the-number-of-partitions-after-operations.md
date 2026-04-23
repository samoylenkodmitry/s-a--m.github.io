---
layout: leetcode-entry
title: "3003. Maximize the Number of Partitions After Operations"
permalink: "/leetcode/problem/2025-10-17-3003-maximize-the-number-of-partitions-after-operations/"
leetcode_ui: true
entry_slug: "2025-10-17-3003-maximize-the-number-of-partitions-after-operations"
---

[3003. Maximize the Number of Partitions After Operations](https://leetcode.com/problems/maximize-the-number-of-partitions-after-operations/) medium
[blog post](https://leetcode.com/problems/maximize-the-number-of-partitions-after-operations/solutions/7281801/kotlin-by-samoylenkodmitry-d9b2/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05102025-3003-maximize-the-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/GedkAHLlp5M)

![fb9e2b9c-0ec1-40cf-995a-41ec74832ea0 (1).webp](/assets/leetcode_daily_images/e789805e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1145

#### Problem TLDR

Max k-uniq parts after changing one letter #hard #prefix

#### Intuition

Didn't solve.

```j
    // accca    k=2
    // 12222
    // abcca
    // 123    cca
    // acbca
    // 123    bca  bc a
    // accba
    // 1223   ba

    // 47 minutes: my algo stuck in corner cases, looking for hints
    // partition_start is not very obvious
    // 56 minute: look for solution

```

Precompute suffix & prefix: parts count and uniqs count so far at `i`.
Heuristic to split into three parts: left and right parts are full and uniqs are not full.
Heuristic to not split: count uniqs is less than k, so letter should go to left or right.
Otherwise split once.

#### Approach

* prefix can be computed on the go

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 9ms
    fun maxPartitionsAfterOperations(s: String, k: Int): Int {
        val sf = Array(s.length) { IntArray(2) }
        var msk = 0; var part = 0; var res = 0
        for (i in s.lastIndex downTo 1) {
            val bit = 1 shl (s[i]-'a'); msk = msk or bit
            if (msk.countOneBits() > k) { part++; msk = bit }
            sf[i-1][0] = part; sf[i-1][1] = msk
        }
        msk = 0; part = 0
        for (i in 0..<s.lastIndex) {
            val cntall = (msk or sf[i][1]).countOneBits()
            res = max(res, part + sf[i][0] +
                if (msk.countOneBits() == k && sf[i][1].countOneBits() == k && cntall < 26) 2
                else if (min(cntall + 1, 26) <= k) 0 else 1)
            val bit = 1 shl (s[i]-'a'); msk = msk or bit
            if (msk.countOneBits() > k) { part++; msk = bit }
        }
        return res + 1
    }

```

