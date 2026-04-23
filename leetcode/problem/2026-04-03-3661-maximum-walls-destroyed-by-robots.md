---
layout: leetcode-entry
title: "3661. Maximum Walls Destroyed by Robots"
permalink: "/leetcode/problem/2026-04-03-3661-maximum-walls-destroyed-by-robots/"
leetcode_ui: true
entry_slug: "2026-04-03-3661-maximum-walls-destroyed-by-robots"
---

[3661. Maximum Walls Destroyed by Robots](https://leetcode.com/problems/maximum-walls-destroyed-by-robots/solutions/7761704/kotlin-by-samoylenkodmitry-v5sh/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05102025-3661-maximum-walls-destroyed?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pmZl5AmQLr0)

![03.04.2026.webp](/assets/leetcode_daily_images/03.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1317

#### Problem TLDR

Max walls hit, each robot shoot left or right #hard #dp

#### Intuition

Didn't solve.
```j
    // 1       4           10
    // <  *  *   *  *  >
    // 1               7
    // 4-3..4+3
    // ranges intersection:
    //     *ab***
    //
    //   ***a*b**
    // count all walls in ranges
    //
    // 0 1 2 3 4 5 6 7 8 9 10
    //   * r *   * * * * * r
    //     w     w   w
    //
    // 27minute wrong answer 523/602 test case
    //
    // so the wrong was my understanding of the problem
    // each robot have to choose which way to shoot, not both
    //
```
Dp state: current robot and should it account for previous robot range.

#### Approach

* sort
* start with top-down dp

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 500ms
    fun maxWalls(r: IntArray, d: IntArray, w: IntArray): Int {
        val x = r.indices.sortedBy { r[it] }; w.sort(); val dp = HashMap<Int, Int>()
        val bs = { v: Int -> w.binarySearch(v).let { max(it.inv(), it) }}
        fun cnt(a: Int, b: Int) = if (a > b) 0 else bs(b+1)-bs(a)
        fun dfs(i: Int, clipLeft: Int): Int = if (i == r.size) 0 else dp.getOrPut(i*d.size*2+clipLeft) {
            val p = r[x[i]]; val rad = d[x[i]]
            val prev = if (i < 1) 0 else if (clipLeft>0) min(r[x[i-1]]+d[x[i-1]],p-1) else r[x[i-1]]
            val L = max(p-rad, prev + 1)
            val R = min(p+rad, if (i+1 < r.size) r[x[i+1]]-1 else Int.MAX_VALUE)
            max(cnt(L, p) + dfs(i+1, 0), cnt(p, R) + dfs(i+1, 1))
        }
        return dfs(0, 0)
    }
```

