---
layout: leetcode-entry
title: "1339. Maximum Product of Splitted Binary Tree"
permalink: "/leetcode/problem/2022-12-10-1339-maximum-product-of-splitted-binary-tree/"
leetcode_ui: true
entry_slug: "2022-12-10-1339-maximum-product-of-splitted-binary-tree"
---

[1339. Maximum Product of Splitted Binary Tree](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/description/) medium

[https://t.me/leetcode_daily_unstoppable/47](https://t.me/leetcode_daily_unstoppable/47)

[blog post](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/solutions/2896607/kotlin-two-dfs/)

```kotlin

    fun maxProduct(root: TreeNode?): Int {
        fun sumDfs(root: TreeNode?): Long {
            return if (root == null) 0L
            else with(root) { `val`.toLong() + sumDfs(left) + sumDfs(right) }
        }
        val total = sumDfs(root)
        fun dfs(root: TreeNode?) : Pair<Long, Long> {
            if (root == null) return Pair(0,0)
            val left = dfs(root.left)
            val right = dfs(root.right)
            val sum = left.first + root.`val`.toLong() + right.first
            val productLeft = left.first * (total - left.first)
            val productRight = right.first * (total - right.first)
            val prevProductMax = maxOf(right.second, left.second)
            return sum to maxOf(productLeft, productRight, prevProductMax)
        }
        return (dfs(root).second % 1_000_000_007L).toInt()
    }

```

Just iterate over all items and compute all products.
We need to compute total sum before making the main traversal.

Space: O(logN), Time: O(N)

