---
layout: leetcode-entry
title: "104. Maximum Depth of Binary Tree"
permalink: "/leetcode/problem/2023-02-16-104-maximum-depth-of-binary-tree/"
leetcode_ui: true
entry_slug: "2023-02-16-104-maximum-depth-of-binary-tree"
---

[104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/) easy

[blog post](https://leetcode.com/problems/maximum-depth-of-binary-tree/solutions/3192288/kotlin-one-liner/)

```kotlin
    fun maxDepth(root: TreeNode?): Int =
        root?.run { 1 + maxOf(maxDepth(left), maxDepth(right)) } ?: 0

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/120
#### Intuition
Do DFS and choose the maximum on each step.

#### Approach
Let's write a one-liner.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(log_2(n))$$

