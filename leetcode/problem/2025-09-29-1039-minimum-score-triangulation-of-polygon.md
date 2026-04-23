---
layout: leetcode-entry
title: "1039. Minimum Score Triangulation of Polygon"
permalink: "/leetcode/problem/2025-09-29-1039-minimum-score-triangulation-of-polygon/"
leetcode_ui: true
entry_slug: "2025-09-29-1039-minimum-score-triangulation-of-polygon"
---

[1039. Minimum Score Triangulation of Polygon](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/description) medium
[blog post](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/solutions/7233233/kotlin-rust-by-samoylenkodmitry-v9ys/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29092025-1039-minimum-score-triangulation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BSaCQ8pGFtA)

![1.webp](/assets/leetcode_daily_images/6a1b99a7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1127

#### Problem TLDR

Min sum of triangle product for every possible triangulation #medium #dp

#### Intuition

Used the hint.

How to triangulate: keep first and last vertice, try every possible third between them.
Memorize for every possible `from` and `to`.

```j
    // 3745
    //
    //    3       7
    //
    //
    //    5       4
    //
    // 375+457 or 345+347
    //
    // 1 3 1 4 1 5
    //
    //       1    3
    //                 1
    //
    //                 4
    //       5     1
    //
    // 113 114 115 111
    //
    // how to triangulate? can't do 111+345
    // maybe recursive? (is it 50^50?)
    // 1 3 1 4 1 5
    // 1 3 1 + 1 1 4 1 5
    // 29 minute: every time we have a ring, just choose the top 2 and split at them
    // but what if they are consequtive?
    //
    // 0 1 2 3 4 5 6
    //   *     *
    // 43 minute (solved for max instead of min)
    // 45 minute (the two vertex algo is not optimal)
    // should we peek 3 vertices?
    // 51 minute: wrong answer 2144
    //
    //  2    1
    //
    //  4    4
    // looks like just picking min values is not enough
    // 54 minute take hints: just a single split?
    // 76 minute TLE (with dp?) [5,80,62,45,96,100,17,72,67,64,20,66,41,68,34,67,35,24,76,2]
    // ^ language syntax error
```

#### Approach

* the hardest part is to find a clever triangulation technique

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

// 58ms
    fun minScoreTriangulation(v: IntArray): Int {
        val l = v.asList(); val dp = HashMap<Pair<Int, Int>, Int>()
        fun d(from: Int, to: Int): Int = if (to-from+1 < 3) 0 else dp.getOrPut(from to to)
            { (from+1..<to).minOf { l[it]*l[from]*l[to] + d(from, it) + d(it, to) } }
        return d(0, v.lastIndex)
    }

```

```rust

// 0ms
    pub fn min_score_triangulation(v: Vec<i32>) -> i32 {
        let mut d = [[0;50];50];
        for i in (0..v.len()).rev() { for j in (i+1..v.len()) { for k in (i+1..j) {
            d[i][j]=(d[i][k]+d[k][j]+v[i]*v[j]*v[k]).min(if d[i][j]==0 {i32::MAX}else {d[i][j]})
        }}}; d[0][v.len()-1]
    }

```

