---
layout: leetcode-entry
title: "783. Minimum Distance Between BST Nodes"
permalink: "/leetcode/problem/2023-02-17-783-minimum-distance-between-bst-nodes/"
leetcode_ui: true
entry_slug: "2023-02-17-783-minimum-distance-between-bst-nodes"
---

[783. Minimum Distance Between BST Nodes](https://leetcode.com/problems/minimum-distance-between-bst-nodes/submissions/899622255/) easy

[blog post](https://leetcode.com/problems/minimum-distance-between-bst-nodes/solutions/3196399/kotlin-morris-traversal/)

```kotlin
    fun minDiffInBST(root: TreeNode?): Int {
        var prev: TreeNode? = null
        var curr = root
        var minDiff = Int.MAX_VALUE
        while (curr != null) {
            if (curr.left == null) {
                if (prev != null) minDiff = minOf(minDiff, Math.abs(curr.`val` - prev.`val`))
                prev = curr
                curr = curr.right
            } else {
                var right = curr.left!!
                while (right.right != null && right.right != curr) right = right.right!!
                if (right.right == curr) {
                    right.right = null
                    curr = curr.right
                } else {
                    right.right = curr
                    val next = curr.left
                    curr.left = null
                    curr = next
                }
            }
        }
        return minDiff
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/121
#### Intuition
Given that this is a Binary Search Tree, `inorder` traversal will give us an increasing sequence of nodes. Minimum difference will be one of the adjacent nodes differences.
#### Approach
Let's write Morris Traversal. Store current node at the rightmost end of the left children.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(1)$$

