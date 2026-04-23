---
layout: leetcode-entry
title: "3129. Find All Possible Stable Binary Arrays I"
permalink: "/leetcode/problem/2026-03-09-3129-find-all-possible-stable-binary-arrays-i/"
leetcode_ui: true
entry_slug: "2026-03-09-3129-find-all-possible-stable-binary-arrays-i"
---

[3129. Find All Possible Stable Binary Arrays I](https://open.substack.com/pub/dmitriisamoilenko/p/09032026-3129-find-all-possible-stable?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/09032026-3129-find-all-possible-stable?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09032026-3129-find-all-possible-stable?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/femYUYMT-gU)

![8a864214-ee89-4c40-85b3-f89e71cb222c (1).webp](/assets/leetcode_daily_images/d43426b0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1292

#### Problem TLDR

Count 01-arrays with z zeros, o ones, at most l repeat #medium

#### Intuition

Didn't solve.
```j
    // length = z+o = 400
    // consequent repeats at most l
    // this can be DFS + memo n^3
    // my solution is n^4 TLE
    // the symmetry trick didn't help
    // lets look hints
    // MLE
```
The working intuition: build arrays by alterating blocks.

#### Approach

* inside DFS: try at most min(l, current) to take, flip the arguments

#### Complexity

- Time complexity:
$$O(zol)$$

- Space complexity:
$$O(zo)$$

#### Code

```kotlin
// 218ms
    fun numberOfStableArrays(z: Int, o: Int, l: Int): Int {
        val dp = HashMap<Int, Int>()
        fun d(z: Int, o: Int): Int =
        if (z == 0) 0 else if (o == 0) {if (z <= l) 1 else 0}
        else dp.getOrPut(z*400+o) {
            (1..min(z,l)).fold(0){ r, nz -> (r+d(o,z-nz))%1000000007}
        }
        return (d(z, o) + d(o, z))%1000000007
    }
```

