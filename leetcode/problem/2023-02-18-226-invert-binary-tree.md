---
layout: leetcode-entry
title: "226. Invert Binary Tree"
permalink: "/leetcode/problem/2023-02-18-226-invert-binary-tree/"
leetcode_ui: true
entry_slug: "2023-02-18-226-invert-binary-tree"
---

[226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/description/) easy

[blog post](https://leetcode.com/problems/invert-binary-tree/solutions/3200281/kotlin-one-liner/)

```kotlin
    fun invertTree(root: TreeNode?): TreeNode? =
        root?.apply { left = invertTree(right).also { right = invertTree(left) } }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/122
#### Intuition
Walk tree with Depth-First Search and swap each left and right nodes.
#### Approach
Let's write a recursive one-liner.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(log_2(n))$$

