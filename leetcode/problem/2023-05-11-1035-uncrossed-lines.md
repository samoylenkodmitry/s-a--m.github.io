---
layout: leetcode-entry
title: "1035. Uncrossed Lines"
permalink: "/leetcode/problem/2023-05-11-1035-uncrossed-lines/"
leetcode_ui: true
entry_slug: "2023-05-11-1035-uncrossed-lines"
---

[1035. Uncrossed Lines](https://leetcode.com/problems/uncrossed-lines/description/) medium

```kotlin

fun maxUncrossedLines(nums1: IntArray, nums2: IntArray): Int {
    val cache = Array(nums1.size) { Array(nums2.size) { -1 } }
    val intersect = nums1.toSet().intersect(nums2.toSet())

    fun dfs(i: Int, j: Int, x: Int): Int {
        if (i == nums1.size || j == nums2.size) return 0
        val cached = cache[i][j]
        if (cached != -1) return cached
        val n1 = nums1[i]
        val n2 = nums2[j]
        val drawLine = if (n1 == x && n2 == x || n1 == n2) 1 + dfs(i + 1, j + 1, n1) else 0
        val skipTop = dfs(i + 1, j, x)
        val skipBottom = dfs(i, j + 1, x)
        val skipBoth = dfs(i + 1, j + 1, x)
        val startTop = if (intersect.contains(n1)) dfs(i + 1, j, n1) else 0
        val startBottom = if (intersect.contains(n2)) dfs(i, j + 1, n2) else 0
        val res = maxOf(
        drawLine,
        maxOf(drawLine, skipTop, skipBottom),
        maxOf(skipBoth, startTop, startBottom)
        )
        cache[i][j] = res
        return res
    }
    return dfs(0, 0, 0)
}

```

[blog post](https://leetcode.com/problems/uncrossed-lines/solutions/3510891/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-11052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/209
#### Intuition
Consider the case:

```

2 5 1 2 5
2 2 2 1 1 1 5 5 5

```

![image.png](/assets/leetcode_daily_images/c7ac81cd.webp)

When we draw all the possible lines, we see that there is a choice to draw line `2-2` or four lines `1-1` or three `5-5` in the middle. Suffix lines `5-5` and prefix lines `2-2` are optimal already and can be cached as a result.
To find an optimal choice we can use DFS.
We can prune some impossible combinations by precomputing the intersected numbers and considering them only.
#### Approach
* use an array for the faster cache instead of HashMap
* for the intersection there is an `intersect` method in Kotlin

#### Complexity
- Time complexity:
$$O(n^3)$$
- Space complexity:
$$O(n^3)$$

