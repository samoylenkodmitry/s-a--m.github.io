---
layout: leetcode-entry
title: "Count Complete Tree Nodes"
permalink: "/leetcode/problem/2022-11-15-count-complete-tree-nodes/"
leetcode_ui: true
entry_slug: "2022-11-15-count-complete-tree-nodes"
---

[https://leetcode.com/problems/count-complete-tree-nodes/](https://leetcode.com/problems/count-complete-tree-nodes/) medium

```

       x
     *   x
   *   *   x
 *   x   *   x
* x x x x * x x
          \
          on each node we can check it's left and right depths
          this only takes us O(logN) time on each step
          there are logN steps in total (height of the tree)
          so the total time complexity is O(log^2(N))

```

```kotlin

    fun countNodes(root: TreeNode?): Int {
        var hl = 0
        var node = root
        while (node != null) {
            node = node.left
            hl++
        }
        var hr = 0
        node = root
        while (node != null) {
            node = node.right
            hr++
        }
        return when {
            hl == 0 -> 0
            hl == hr -> (1 shl hl) - 1
            else -> 1  +
            (root!!.left?.let {countNodes(it)}?:0) +
            (root!!.right?.let {countNodes(it)}?:0)
        }
    }

```

Complexity: O(log^2(N))
Memory: O(logN)

