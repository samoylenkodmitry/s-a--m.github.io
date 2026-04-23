---
layout: leetcode-entry
title: "1026. Maximum Difference Between Node and Ancestor"
permalink: "/leetcode/problem/2022-12-09-1026-maximum-difference-between-node-and-ancestor/"
leetcode_ui: true
entry_slug: "2022-12-09-1026-maximum-difference-between-node-and-ancestor"
---

[1026. Maximum Difference Between Node and Ancestor](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/description/) medium

[https://t.me/leetcode_daily_unstoppable/46](https://t.me/leetcode_daily_unstoppable/46)

[blog post](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/solutions/2894948/kotlin-dfs/)

```kotlin

    fun maxAncestorDiff(root: TreeNode?): Int {
        root?: return 0

        fun dfs(root: TreeNode, min: Int = root.`val`, max: Int = root.`val`): Int {
            val v = root.`val`
            val currDiff = maxOf(Math.abs(v - min), Math.abs(v - max))
            val currMin = minOf(min, v)
            val currMax = maxOf(max, v)
            val leftDiff = root.left?.let { dfs(it, currMin, currMax) } ?: 0
            val rightDiff = root.right?.let { dfs(it, currMin, currMax) } ?: 0
            return maxOf(currDiff, leftDiff, rightDiff)
        }

        return dfs(root)
    }

```

Based on math we can assume, that max difference is one of the two: (curr - max so far) or (curr - min so far).
Like, for example, let our curr value be `3`, and from all visited we have min `0` and max `7`.

```

 0--3---7

```

* we can write helper recoursive method and compute max and min so far

Space: O(logN), Time: O(N)

