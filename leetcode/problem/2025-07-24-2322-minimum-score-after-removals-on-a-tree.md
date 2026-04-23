---
layout: leetcode-entry
title: "2322. Minimum Score After Removals on a Tree"
permalink: "/leetcode/problem/2025-07-24-2322-minimum-score-after-removals-on-a-tree/"
leetcode_ui: true
entry_slug: "2025-07-24-2322-minimum-score-after-removals-on-a-tree"
---

[2322. Minimum Score After Removals on a Tree](https://leetcode.com/problems/minimum-score-after-removals-on-a-tree/description) hard
[blog post](https://leetcode.com/problems/minimum-score-after-removals-on-a-tree/solutions/6997956/kotlin-by-samoylenkodmitry-hfsl/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24072025-2322-minimum-score-after?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xhHGnqQa34w)
![1.webp](/assets/leetcode_daily_images/b34effce.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1059

#### Problem TLDR

Min(max_group_xor - min_group_xor) by removing 2 edges from tree #hard #uf

#### Intuition

Didn't solved.

```j

   // xor(all) = X
    // xor(B, C) = X xor A
    // A = B xor C xor X
    // min = max xor C xor X
    // score = max - min = max - (max xor C xor X)
    // I need max score
    // score = A - A^C^X, A is max
    // X = A^B^C
    //     A is max
    //       B is min
    //         C is X^max^min
    // A - B =
    //
    // score^X = max^X - max^C
    //
    // going circles with xor arithmetics
    //
    // xor can decrease: 1 xor 1 = 0
    // xor can increase: 0 xor 1 = 1
    // 101^10 - 101^01 = 110 - 100 = 100
    // ok forget about math, what about tree walk
    // how to disconnect edge?
    // 25 minutes, use hint
    // first split a single edge
    // A vs BC
    // then choose the best second split point (how? 32 minute)
    // if A is min, then look for max in BC
    // if A is max, then look for min in BC
    // if A is neutral - then it is irrelevant, we will traverse all possible A anyway
    // ...............
    // ..........A____B
    // _____B.........A
    // for each edge we have a pair A-BC
    //                              max(max_A) - min(min_A) ? not that simple
    // 50 minute gave up, gosh I even forgot we have to find the MINIMUM (max-min)

```

My intuition direction was on Union-Find, but I had a hard time computing the xor cases.

Some stolen solution intuition:

* precompute Union-Find results for all single edges: group `roots`, and two values for `(xorA, xorBC)`
* then again iterate `i, j` and calculate 3 xors based on `logic`

The tricky part, `logic` and I still have a hard time to really get it.
* on value we always take, let it be `c2 = A`
* we have `4 nodes` of `2` disconnected edges `(a-/-b), (c-/-d)`
* then the part I don't fully get, look at the source (https://leetcode.com/problems/minimum-score-after-removals-on-a-tree/solutions/2199132/union-find-c-o-n-2-time-o-n-2-space-code-explanation/)

#### Approach

* don't spend too much time hitting a head against the wall, but at least read, something will be absorbed anyway

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

// 269ms
    fun minimumScore(n: IntArray, es: Array<IntArray>): Int {
        val us = Array(n.size) { IntArray(n.size) { it }}; val xs = Array(n.size) { IntArray(2) }
        operator fun IntArray.div(x: Int): Int = if (get(x) == x) x else (this/get(x)).also { set(x, it) }
        for (a in es.indices) {
            val u = us[a]; val r = IntArray(n.size) { n[it] }
            for (i in es.indices) if (i != a) {
                val a = u/es[i][0]; val b = u/es[i][1]
                if (a != b) { u[a] = b; r[b] = r[b] xor r[a]; r[a] = 0 }
            }
            xs[a][0] = r[u/es[a][0]]; xs[a][1] = r[u/es[a][1]]; us[a] = u
        }
        var res = Int.MAX_VALUE
        for (i in es.indices) for (j in es.indices) if (i != j) {
            val (a, b) = es[i]; var (c, d) = es[j]; var c1 = 0; var c2 = 0; var c3 = 0
            if (us[i]/a == us[i]/c) {
                c1 = if (us[j]/a == us[j]/c) xs[j][1] else xs[j][0]
                c2 = xs[i][1]; c3 = c1 xor xs[i][0]
            } else {
                c1 = if (us[j]/b == us[j]/c) xs[j][1] else xs[j][0]
                c2 = xs[i][0]; c3 = c1 xor xs[i][1]
            }
            res = min(res, maxOf(c1, c2, c3) - minOf(c1, c2, c3))
        }
        return res
    }

```

