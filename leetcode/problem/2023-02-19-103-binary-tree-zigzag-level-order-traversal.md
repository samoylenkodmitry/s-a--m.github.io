---
layout: leetcode-entry
title: "103. Binary Tree Zigzag Level Order Traversal"
permalink: "/leetcode/problem/2023-02-19-103-binary-tree-zigzag-level-order-traversal/"
leetcode_ui: true
entry_slug: "2023-02-19-103-binary-tree-zigzag-level-order-traversal"
---

[103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/) medium

[blog post](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/solutions/3204919/kotlin-bfs/)

```kotlin
    fun zigzagLevelOrder(root: TreeNode?): List<List<Int>> = mutableListOf<List<Int>>().also { res ->
            with(ArrayDeque<TreeNode>().apply { root?.let { add(it) } }) {
                while (isNotEmpty()) {
                    val curr = LinkedList<Int>().apply { res.add(this) }
                    repeat(size) {
                        with(poll()) {
                            with(curr) { if (res.size % 2 == 0) addFirst(`val`) else addLast(`val`) }
                            left?.let { add(it) }
                            right?.let { add(it) }
                        }
                    }
                }
            }
        }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/123
#### Intuition
Each BFS step gives us a level, which one we can reverse if needed.

#### Approach
* for zigzag, we can skip a boolean variable and track result count.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$

