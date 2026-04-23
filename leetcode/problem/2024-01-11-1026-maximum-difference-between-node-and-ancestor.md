---
layout: leetcode-entry
title: "1026. Maximum Difference Between Node and Ancestor"
permalink: "/leetcode/problem/2024-01-11-1026-maximum-difference-between-node-and-ancestor/"
leetcode_ui: true
entry_slug: "2024-01-11-1026-maximum-difference-between-node-and-ancestor"
---

[1026. Maximum Difference Between Node and Ancestor](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/description/) medium
[blog post](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/solutions/4544360/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11012024-1026-maximum-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/0ZbZ7yV4gY8)
![image.png](/assets/leetcode_daily_images/6f1a01d0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/468

#### Problem TLDR

Max diff between node and ancestor in a binary tree.

#### Intuition

Let's traverse the tree with Depth-First Search and keep track of the max and min values.

#### Approach

* careful with corner case: min and max must be in the same ancestor-child hierarchy
* we can use external variable, or put it in each result

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

  fun maxAncestorDiff(root: TreeNode?): Int {
    var res = 0
    fun dfs(n: TreeNode?): List<Int> = n?.run {
      (dfs(left) + dfs(right) + listOf(`val`)).run {
        listOf(min(), max()).onEach { res = max(res, abs(`val` - it)) }
      }
    } ?: listOf()
    dfs(root)
    return res
  }

```

