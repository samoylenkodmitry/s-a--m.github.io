---
layout: leetcode-entry
title: "111. Minimum Depth of Binary Tree"
permalink: "/leetcode/problem/2023-07-11-111-minimum-depth-of-binary-tree/"
leetcode_ui: true
entry_slug: "2023-07-11-111-minimum-depth-of-binary-tree"
---

[111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/description/) easy
[blog post](https://leetcode.com/problems/minimum-depth-of-binary-tree/solutions/3743369/kotlin-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/11072023-111-minimum-depth-of-binary?sd=pf)
![image.png](/assets/leetcode_daily_images/1d163b53.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/271
#### Problem TLDR
Count nodes in the shortest path from root to leaf
#### Intuition
* remember to count `nodes`, not `edges`
* `leaf` is a node without children
* use BFS or DFS

#### Approach
Let's use BFS

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun minDepth(root: TreeNode?): Int = with(ArrayDeque<TreeNode>()) {
    root?.let { add(it) }
    generateSequence(1) { (it + 1).takeIf { isNotEmpty() } }
    .firstOrNull {
        (1..size).any {
            with(poll()) {
                left?.let { add(it) }
                right?.let { add(it) }
                left == null && right == null
            }
        }
    } ?: 0
}

```

