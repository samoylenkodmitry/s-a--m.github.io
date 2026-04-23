---
layout: leetcode-entry
title: "662. Maximum Width of Binary Tree"
permalink: "/leetcode/problem/2023-04-20-662-maximum-width-of-binary-tree/"
leetcode_ui: true
entry_slug: "2023-04-20-662-maximum-width-of-binary-tree"
---

[662. Maximum Width of Binary Tree](https://leetcode.com/problems/maximum-width-of-binary-tree/description/) medium

```kotlin

fun widthOfBinaryTree(root: TreeNode?): Int =
with(ArrayDeque<Pair<Int, TreeNode>>()) {
    root?.let { add(0 to it) }
    var width = 0
    while (isNotEmpty()) {
        var first = peek()
        var last = last()
        width = maxOf(width, last.first - first.first + 1)
        repeat(size) {
            val (x, node) = poll()
            node.left?.let { add(2 * x + 1 to it) }
            node.right?.let { add(2 * x + 2 to it) }
        }
    }
    width
}

```

[blog post](https://leetcode.com/problems/maximum-width-of-binary-tree/solutions/3436856/kotlin-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-20042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/186
#### Intuition
For every node, positions of it's left child is $$2x +1$$ and right is $$2x + 2$$
![leetcode_tree.gif](/assets/leetcode_daily_images/c5230258.webp)

#### Approach
We can do BFS and track node positions.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

