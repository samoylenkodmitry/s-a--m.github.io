---
layout: leetcode-entry
title: "872. Leaf-Similar Trees"
permalink: "/leetcode/problem/2022-12-08-872-leaf-similar-trees/"
leetcode_ui: true
entry_slug: "2022-12-08-872-leaf-similar-trees"
---

[872. Leaf-Similar Trees](https://leetcode.com/problems/leaf-similar-trees/solutions/) easy

[https://t.me/leetcode_daily_unstoppable/45](https://t.me/leetcode_daily_unstoppable/45)

```kotlin

    fun leafSimilar(root1: TreeNode?, root2: TreeNode?): Boolean {
        fun dfs(root: TreeNode?): List<Int> {
            return when {
                root == null -> listOf()
                root.left == null && root.right == null -> listOf(root.`val`)
                else -> dfs(root.left) + dfs(root.right)
            }
        }

        return dfs(root1) == dfs(root2)
    }

```

There is only 200 items, so we can concatenate lists.
One optimization would be to collect only first tree and just compare it to the second tree while doing the inorder traverse.

Space: O(N), Time: O(N)

