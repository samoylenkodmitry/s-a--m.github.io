---
layout: leetcode-entry
title: "3363. Find the Maximum Number of Fruits Collected"
permalink: "/leetcode/problem/2025-08-07-3363-find-the-maximum-number-of-fruits-collected/"
leetcode_ui: true
entry_slug: "2025-08-07-3363-find-the-maximum-number-of-fruits-collected"
---

[3363. Find the Maximum Number of Fruits Collected](https://leetcode.com/problems/find-the-maximum-number-of-fruits-collected/description) hard
[blog post](https://leetcode.com/problems/find-the-maximum-number-of-fruits-collected/solutions/7054079/kotlin-by-samoylenkodmitry-xgta/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/7082025-3363-find-the-maximum-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ogqlVLt0qxA)
![1.webp](/assets/leetcode_daily_images/91496675.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1073

#### Problem TLDR

Max 3 paths from corners to bottom end #hard

#### Intuition

Used the hints.

Two insights:
1. middle only goes diagonal
2. two other guys are the two separate searches

```j
    // dfs or bfs?
    // greedy? - i think can be non-optimal
    // either full search with bfs or dp with dfs+cache
    // 3x3x3 = 27 steps by each bfs layer
    // exactly n-1 steps AND reach n,n cell - no non-optimal steps
    // i am writing the dfs+dp, the growth factor is n^27
    // let's use hints
    // nice, child 0 can only move diagonal (as by rules)
    // child 1&2 can't cross diagonal
    // my solution TLE
    // look for other hints, no new information
    // are we expected use raw arrays? bottom up?
    // expected bottom up
    // children b and c can go separate
```

#### Approach

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

// 697ms
    fun maxCollectedFruits(f: Array<IntArray>, n: Int = f.lastIndex): Int {
        data class P(val x: Int, val y: Int) { val i = x < 0 || y < 0 || x > n || y > n }
        fun dfs(p: P, d: List<Int>, dp: HashMap<P, Int> = HashMap()): Int = dp.getOrPut(p) {
            f[p.y][p.x] + (0..2).maxOf {
                val b = P(p.x + d[it*2], p.y + d[it*2 + 1])
                if (!b.i && (b.y - b.x).sign == d[2].sign) dfs(b, d, dp) else 0
            }}
        val b = listOf(1, 1, -1, 1, 0, 1); val c = listOf(1, 0, 1, 1, 1, -1)
        return dfs(P(n, 0), b) + dfs(P(0, n), c) + (0..n).sumOf { f[it][it] }
    }

```

