---
layout: leetcode-entry
title: "1232. Check If It Is a Straight Line"
permalink: "/leetcode/problem/2023-06-05-1232-check-if-it-is-a-straight-line/"
leetcode_ui: true
entry_slug: "2023-06-05-1232-check-if-it-is-a-straight-line"
---

[1232. Check If It Is a Straight Line](https://leetcode.com/problems/check-if-it-is-a-straight-line/description/) easy
[blog post](https://leetcode.com/problems/check-if-it-is-a-straight-line/solutions/3598943/kotlin-tan/)
[substack](https://dmitriisamoilenko.substack.com/p/05062023-1232-check-if-it-is-a-straight?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/236
#### Problem TLDR
Are all the `x,y` points in a line?
#### Intuition
We can compare $$tan_i = dy_i/dx_i = dy_0/dx_0$$

#### Approach
* corner case is a vertical line
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun checkStraightLine(coordinates: Array<IntArray>): Boolean =
    with((coordinates[1][1] - coordinates[0][1])/
    (coordinates[1][0] - coordinates[0][0]).toDouble()) {
        coordinates.drop(2).all {
            val o = (it[1] - coordinates[0][1]) / (it[0] - coordinates[0][0]).toDouble()

            isInfinite() && o.isInfinite() || this == o
        }
    }

```

