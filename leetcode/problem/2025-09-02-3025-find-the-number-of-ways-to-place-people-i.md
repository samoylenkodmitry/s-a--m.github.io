---
layout: leetcode-entry
title: "3025. Find the Number of Ways to Place People I"
permalink: "/leetcode/problem/2025-09-02-3025-find-the-number-of-ways-to-place-people-i/"
leetcode_ui: true
entry_slug: "2025-09-02-3025-find-the-number-of-ways-to-place-people-i"
---

[3025. Find the Number of Ways to Place People I](https://leetcode.com/problems/find-the-number-of-ways-to-place-people-i/description/) medium
[blog post](https://leetcode.com/problems/find-the-number-of-ways-to-place-people-i/solutions/7147340/kotlin-by-samoylenkodmitry-8wm7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02092025-3025-find-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pMiePyA3pbs)

![1.webp](/assets/leetcode_daily_images/a60afc18.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1100

#### Problem TLDR

Left-top, bottom-right pairs with empty rectangles #medium

#### Intuition

Brute-force is accepted for n=50.

Another intuition: sort by `x` then rotate CCW around each point.

#### Approach

* should we learn quad trees?

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 55ms
    fun numberOfPairs(p: Array<IntArray>): Int {
        var res = 0
        for (i in p.indices) for (j in p.indices)
            if (i != j && p[i][0] <= p[j][0] && p[i][1] >= p[j][1] &&
                p.indices.none { k -> k != i && k != j &&
                    p[k][0] in p[i][0]..p[j][0] && p[k][1] in p[j][1]..p[i][1] }) ++res
        return res
    }

```
```kotlin

// 43ms
    fun numberOfPairs(p: Array<IntArray>): Int {
        p.sortBy { it[0]*1000-it[1] }
        return p.indices.sumOf { i ->
            var y = -1
            p.drop(i+1).count { (_,d) -> d <= p[i][1] && d > y.also{y=max(y,d)}}
        }
    }

```

