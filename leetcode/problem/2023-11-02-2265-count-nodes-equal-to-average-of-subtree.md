---
layout: leetcode-entry
title: "2265. Count Nodes Equal to Average of Subtree"
permalink: "/leetcode/problem/2023-11-02-2265-count-nodes-equal-to-average-of-subtree/"
leetcode_ui: true
entry_slug: "2023-11-02-2265-count-nodes-equal-to-average-of-subtree"
---

[2265. Count Nodes Equal to Average of Subtree](https://leetcode.com/problems/count-nodes-equal-to-average-of-subtree/description/) medium
[blog post](https://leetcode.com/problems/count-nodes-equal-to-average-of-subtree/solutions/4237610/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02112023-2265-count-nodes-equal-to?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/34cf7f49.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/390

#### Problem TLDR

Number of nodes in a tree where `val == sum / count` of a subtree

#### Intuition

Just do a Depth First Search and return `sum` and `count` of a subtree.

#### Approach

* avoid nulls when traversing the tree

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$ for the recursion depth

#### Code

```kotlin

    fun averageOfSubtree(root: TreeNode?): Int {
      var res = 0
      fun dfs(n: TreeNode): Pair<Int, Int> {
        val (ls, lc) = n.left?.let { dfs(it) } ?: 0 to 0
        val (rs, rc) = n.right?.let { dfs(it) } ?: 0 to 0
        val sum = n.`val` + ls + rs
        val count = 1 + lc + rc
        if (n.`val` == sum / count) res++
        return sum to count
      }
      root?.let { dfs(it) }
      return res
    }

```

