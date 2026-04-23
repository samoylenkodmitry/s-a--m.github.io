---
layout: leetcode-entry
title: "3027. Find the Number of Ways to Place People II"
permalink: "/leetcode/problem/2025-09-03-3027-find-the-number-of-ways-to-place-people-ii/"
leetcode_ui: true
entry_slug: "2025-09-03-3027-find-the-number-of-ways-to-place-people-ii"
---

[3027. Find the Number of Ways to Place People II](https://leetcode.com/problems/find-the-number-of-ways-to-place-people-ii/description/) hard
[blog post](https://leetcode.com/problems/find-the-number-of-ways-to-place-people-ii/solutions/7150546/kotlin-by-samoylenkodmitry-sbem/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03092025-3027-find-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/th1GCmMw56A)

![1.webp](/assets/leetcode_daily_images/601f7633.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1101

#### Problem TLDR

Left-top, bottom-right pairs with empty rectangles #medium #geometry

#### Intuition

Sort by `x` then rotate CCW around each point.

#### Approach

* attention to the numbers range -10^9..10^9

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 692ms
    fun numberOfPairs(p: Array<IntArray>): Int {
        p.sortWith(compareBy({it[0]},{-it[1]}))
        return p.indices.sumOf { i -> var m = Int.MIN_VALUE
            p.drop(i+1).count { (_,y) -> y <= p[i][1] && y > m.also{m=max(m,y)}}
        }
    }

```

