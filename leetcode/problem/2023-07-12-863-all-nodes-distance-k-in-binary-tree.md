---
layout: leetcode-entry
title: "863. All Nodes Distance K in Binary Tree"
permalink: "/leetcode/problem/2023-07-12-863-all-nodes-distance-k-in-binary-tree/"
leetcode_ui: true
entry_slug: "2023-07-12-863-all-nodes-distance-k-in-binary-tree"
---

[863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/solutions/3748155/kotlin-dfs-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/12072023-863-all-nodes-distance-k?sd=pf)
![image.png](/assets/leetcode_daily_images/540ccb28.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/272
#### Problem TLDR
List of `k` distanced from `target` nodes in a Binary Tree
#### Intuition
There is a one-pass DFS solution, but it feels like too much of a corner cases and result handholding.
A more robust way is to traverse with DFS and connect children nodes to parent, then send a wave from target at `k` steps.

#### Approach
Let's build an undirected graph and do BFS.
* don't forget a visited `HashSet`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun distanceK(root: TreeNode?, target: TreeNode?, k: Int): List<Int> {
    val fromTo = mutableMapOf<Int, MutableList<Int>>()
        fun dfs(node: TreeNode?, parent: TreeNode?) {
            node?.run {
                parent?.let {
                    fromTo.getOrPut(`val`) { mutableListOf() } += it.`val`
                    fromTo.getOrPut(it.`val`) { mutableListOf() } += `val`
                }
                dfs(left, this)
                dfs(right, this)
            }
        }
        dfs(root, null)
        return LinkedList<Int>().apply {
            val visited = HashSet<Int>()
                target?.run {
                    add(`val`)
                    visited.add(`val`)
                }
                repeat(k) {
                    repeat(size) {
                        fromTo.remove(poll())?.forEach { if (visited.add(it)) add(it) }
                    }
                }
            }
        }

```

