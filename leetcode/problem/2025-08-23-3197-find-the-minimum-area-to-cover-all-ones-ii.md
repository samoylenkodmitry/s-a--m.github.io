---
layout: leetcode-entry
title: "3197. Find the Minimum Area to Cover All Ones II"
permalink: "/leetcode/problem/2025-08-23-3197-find-the-minimum-area-to-cover-all-ones-ii/"
leetcode_ui: true
entry_slug: "2025-08-23-3197-find-the-minimum-area-to-cover-all-ones-ii"
---

[3197. Find the Minimum Area to Cover All Ones II](https://leetcode.com/problems/find-the-minimum-area-to-cover-all-ones-ii/description/) hard
[blog post](https://leetcode.com/problems/find-the-minimum-area-to-cover-all-ones-ii/solutions/7112959/kotlin-by-samoylenkodmitry-rfeh/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23082025-3197-find-the-minimum-area?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XpPIkVrLWUE)

![1.webp](/assets/leetcode_daily_images/48b3c8eb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1090

#### Problem TLDR

Min sum all-1 areas of 3-split #hard

#### Intuition

Use hint: try every split

#### Approach

* corner case: consider reverse split and sum

#### Complexity

- Time complexity:
$$O(nm^4)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 51ms
    fun minimumSum(g: Array<IntArray>): Int {
        val w = g[0].size; val h = g.size
        fun sum(x1: Int, y1: Int, x2: Int, y2: Int): Int {
            var r = 0; var b = 0; var l = w; var t = h
            for (y in y1..y2) for (x in x1..x2) if (g[y][x] > 0)
            { r = max(r, x); l = min(l, x); b = max(b, y); t = min(t, y) }
            return (r-l+1)*(b-t+1)
        }
        fun split(x1: Int, y1: Int, x2: Int, y2: Int): Int {
            var res = 30*30
            for (y in y1..<y2) res = min(res, sum(x1, y1, x2, y) + sum(x1, y+1, x2, y2))
            for (x in x1..<x2) res = min(res, sum(x1, y1, x, y2) + sum(x+1, y1, x2, y2))
            return res
        }
        var res = 30*30
        for (y in 0..<h-1) res = min(res, sum(0, 0, w-1, y) + split(0, y+1, w-1, h-1))
        for (y in 0..<h-1) res = min(res, split(0, 0, w-1, y) + sum(0, y+1, w-1, h-1))
        for (x in 0..<w-1) res = min(res, sum(0, 0, x, h-1) + split(x+1, 0, w-1, h-1))
        for (x in 0..<w-1) res = min(res, split(0, 0, x, h-1) + sum(x+1, 0, w-1, h-1))
        return res
    }

```

