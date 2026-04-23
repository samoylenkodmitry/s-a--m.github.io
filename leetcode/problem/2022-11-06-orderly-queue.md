---
layout: leetcode-entry
title: "Orderly Queue"
permalink: "/leetcode/problem/2022-11-06-orderly-queue/"
leetcode_ui: true
entry_slug: "2022-11-06-orderly-queue"
---

[https://leetcode.com/problems/orderly-queue/](https://leetcode.com/problems/orderly-queue/) hard

```kotlin

    fun orderlyQueue(s: String, k: Int): String {
        val chrs = s.toCharArray()
        if (k == 1) {
            var smallest = s
            for (i in 0..s.lastIndex) {
                val prefix = s.substring(0, i)
                val suffix = s.substring(i)
                val ss = suffix + prefix
                if (ss.compareTo(smallest) < 0) smallest = ss
            }
            return smallest
        } else {
            chrs.sort()
            return String(chrs)
        }
    }

O(n^2)

```

Explanation:
One idea that come to my mind is: if k >= 2 then you basically can swap any adjacent elements. That means you can actually sort all the characters.

Speed: O(n^2), Memory: O(n)

