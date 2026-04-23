---
layout: leetcode-entry
title: "124. Binary Tree Maximum Path Sum"
permalink: "/leetcode/problem/2022-12-11-124-binary-tree-maximum-path-sum/"
leetcode_ui: true
entry_slug: "2022-12-11-124-binary-tree-maximum-path-sum"
---

[124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/) hard

[https://t.me/leetcode_daily_unstoppable/48](https://t.me/leetcode_daily_unstoppable/48)

[blog post](https://leetcode.com/problems/binary-tree-maximum-path-sum/solutions/2900498/kotlin-very-bad-problem-definition/)

```kotlin
    fun maxPathSum(root: TreeNode?): Int {
        fun dfs(root: TreeNode): Pair<Int, Int> {
            val lt = root.left
            val rt = root.right
            if (lt == null && rt == null) return root.`val` to root.`val`
            if (lt == null || rt == null) {
                val sub = dfs(if (lt == null) rt else lt)
                val currRes = root.`val` + sub.second
                val maxRes = maxOf(sub.first, currRes, root.`val`)
                val maxPath = maxOf(root.`val`, root.`val` + sub.second)
                return maxRes to maxPath
            } else {
                val left = dfs(root.left)
                val right = dfs(root.right)
                val currRes1 = root.`val` + left.second + right.second
                val currRes2 = root.`val`
                val currRes3 = root.`val` + left.second
                val currRes4 = root.`val` + right.second
                val max1 = maxOf(currRes1, currRes2)
                val max2 = maxOf(currRes3, currRes4)
                val maxRes = maxOf(left.first, right.first, maxOf(max1, max2))
                val maxPath = maxOf(root.`val`, root.`val` + maxOf(left.second, right.second))
                return maxRes to maxPath
            }
        }
        return if (root == null) 0 else dfs(root).first
    }

```

Space: O(logN), Time: O(N)

